from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
import os
import sys

from .loader import deep_merge, load_mapping


SUPPORTED_BROWSERS = ("chrome", "edge", "firefox", "brave", "vivaldi", "opera")
AUTO_BROWSER_VALUES = {"", "auto", "default", "default_browser"}


@dataclass(frozen=True, slots=True)
class BrowserCookieCandidate:
    browser: str
    profile: str = ""
    source: str = "detected"
    reason: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def normalize_browser_name(value: str) -> str:
    lowered = value.strip().lower()
    aliases = {
        "msedge": "edge",
        "microsoft-edge": "edge",
        "microsoft edge": "edge",
        "google-chrome": "chrome",
        "google chrome": "chrome",
        "brave-browser": "brave",
        "mozilla firefox": "firefox",
    }
    return aliases.get(lowered, lowered)


def browser_from_user_choice_progid(progid: str) -> str:
    lowered = progid.strip().lower()
    if "firefox" in lowered:
        return "firefox"
    if "brave" in lowered:
        return "brave"
    if "vivaldi" in lowered:
        return "vivaldi"
    if "opera" in lowered:
        return "opera"
    if "msedge" in lowered or "edge" in lowered:
        return "edge"
    if "chrome" in lowered:
        return "chrome"
    return ""


def detect_default_browser() -> BrowserCookieCandidate | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        import winreg
    except ImportError:
        return None
    for scheme in ("https", "http"):
        key_path = rf"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\{scheme}\UserChoice"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
                progid, _ = winreg.QueryValueEx(key, "ProgId")
        except OSError:
            continue
        browser = browser_from_user_choice_progid(str(progid))
        if browser:
            return BrowserCookieCandidate(browser=browser, source="default_browser", reason=f"{scheme} ProgId={progid}")
    return None


def installed_browser_candidates(*, home: Path | None = None, env: Mapping[str, str] | None = None) -> list[BrowserCookieCandidate]:
    env_map = env or os.environ
    home_dir = home or Path.home()
    candidates: list[BrowserCookieCandidate] = []
    if sys.platform.startswith("win"):
        local = Path(env_map.get("LOCALAPPDATA") or home_dir / "AppData" / "Local")
        roaming = Path(env_map.get("APPDATA") or home_dir / "AppData" / "Roaming")
        probes = {
            "chrome": local / "Google" / "Chrome" / "User Data",
            "edge": local / "Microsoft" / "Edge" / "User Data",
            "brave": local / "BraveSoftware" / "Brave-Browser" / "User Data",
            "vivaldi": local / "Vivaldi" / "User Data",
            "opera": roaming / "Opera Software" / "Opera Stable",
            "firefox": roaming / "Mozilla" / "Firefox" / "Profiles",
        }
    elif sys.platform == "darwin":
        library = home_dir / "Library" / "Application Support"
        probes = {
            "chrome": library / "Google" / "Chrome",
            "edge": library / "Microsoft Edge",
            "brave": library / "BraveSoftware" / "Brave-Browser",
            "vivaldi": library / "Vivaldi",
            "opera": library / "com.operasoftware.Opera",
            "firefox": library / "Firefox" / "Profiles",
        }
    else:
        config_home = Path(env_map.get("XDG_CONFIG_HOME") or home_dir / ".config")
        probes = {
            "chrome": config_home / "google-chrome",
            "edge": config_home / "microsoft-edge",
            "brave": config_home / "BraveSoftware" / "Brave-Browser",
            "vivaldi": config_home / "vivaldi",
            "opera": config_home / "opera",
            "firefox": home_dir / ".mozilla" / "firefox",
        }
    for browser, path in probes.items():
        if path.exists():
            candidates.append(BrowserCookieCandidate(browser=browser, source="installed_browser", reason=str(path)))
    return candidates


def browser_cookie_candidates(*, browser_hint: str = "") -> list[BrowserCookieCandidate]:
    candidates: list[BrowserCookieCandidate] = []
    seen: set[tuple[str, str]] = set()

    def add(candidate: BrowserCookieCandidate | None) -> None:
        if candidate is None:
            return
        key = (candidate.browser, candidate.profile)
        if candidate.browser not in SUPPORTED_BROWSERS or key in seen:
            return
        seen.add(key)
        candidates.append(candidate)

    hint = normalize_browser_name(browser_hint)
    if hint in SUPPORTED_BROWSERS:
        add(BrowserCookieCandidate(browser=hint, source="browser_hint", reason="provided by caller"))
    add(detect_default_browser())
    for candidate in installed_browser_candidates():
        add(candidate)
    return candidates


def resolve_browser_cookie_candidate(browser: str = "auto", profile: str = "") -> BrowserCookieCandidate | None:
    normalized = normalize_browser_name(browser)
    if normalized not in AUTO_BROWSER_VALUES:
        if normalized not in SUPPORTED_BROWSERS:
            return None
        return BrowserCookieCandidate(browser=normalized, profile=profile.strip(), source="explicit", reason="provided by caller")
    candidates = browser_cookie_candidates()
    if not candidates:
        return None
    first = candidates[0]
    if profile.strip():
        return BrowserCookieCandidate(browser=first.browser, profile=profile.strip(), source=first.source, reason=first.reason)
    return first


def youtube_cookie_config_patch(candidate: BrowserCookieCandidate) -> dict:
    return {
        "youtube": {
            "cookies_from_browser": candidate.browser,
            "cookies_browser_profile": candidate.profile,
        },
        "source": {
            "cookies_from_browser": candidate.browser,
            "cookies_browser_profile": candidate.profile,
        },
    }


def apply_youtube_cookie_config(
    config_path: Path,
    *,
    browser: str = "auto",
    profile: str = "",
    dry_run: bool = False,
) -> dict:
    candidate = resolve_browser_cookie_candidate(browser, profile)
    if candidate is None:
        return {
            "updated": False,
            "config_path": str(config_path),
            "error": f"No supported browser cookie source found for {browser!r}.",
            "candidates": [item.to_dict() for item in browser_cookie_candidates()],
        }
    patch = youtube_cookie_config_patch(candidate)
    config_path = config_path.expanduser().resolve()
    if not dry_run:
        current = load_mapping(config_path)
        updated = deep_merge(current, patch)
        write_mapping(config_path, updated)
    return {
        "updated": not dry_run,
        "dry_run": dry_run,
        "config_path": str(config_path),
        "candidate": candidate.to_dict(),
        "patch": patch,
    }


def write_mapping(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    except ImportError:
        import json

        text = json.dumps(data, ensure_ascii=False, indent=2)
    path.write_text(text, encoding="utf-8")
