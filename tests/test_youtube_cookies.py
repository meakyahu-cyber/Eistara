from __future__ import annotations

import sys
from pathlib import Path

from eistara.config.loader import load_mapping
from eistara.config.youtube_cookies import (
    apply_youtube_cookie_config,
    BrowserCookieCandidate,
    browser_from_user_choice_progid,
    installed_browser_candidates,
    resolve_browser_cookie_candidate,
)


def test_browser_from_user_choice_progid_maps_common_windows_browsers() -> None:
    assert browser_from_user_choice_progid("ChromeHTML") == "chrome"
    assert browser_from_user_choice_progid("MSEdgeHTM") == "edge"
    assert browser_from_user_choice_progid("FirefoxURL-308046B0AF4A39CB") == "firefox"


def test_installed_browser_candidates_finds_windows_profiles(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    local = tmp_path / "Local"
    roaming = tmp_path / "Roaming"
    (local / "Microsoft" / "Edge" / "User Data").mkdir(parents=True)
    (roaming / "Mozilla" / "Firefox" / "Profiles").mkdir(parents=True)

    candidates = installed_browser_candidates(
        home=tmp_path,
        env={"LOCALAPPDATA": str(local), "APPDATA": str(roaming)},
    )

    assert [candidate.browser for candidate in candidates] == ["edge", "firefox"]


def test_resolve_browser_cookie_candidate_accepts_explicit_browser() -> None:
    candidate = resolve_browser_cookie_candidate("microsoft edge", "Default")

    assert candidate is not None
    assert candidate.browser == "edge"
    assert candidate.profile == "Default"


def test_resolve_browser_cookie_candidate_auto_prefers_default_browser(monkeypatch) -> None:
    monkeypatch.setattr(
        "eistara.config.youtube_cookies.detect_default_browser",
        lambda: BrowserCookieCandidate("firefox", source="default_browser"),
    )
    monkeypatch.setattr(
        "eistara.config.youtube_cookies.installed_browser_candidates",
        lambda: [BrowserCookieCandidate("edge", source="installed_browser")],
    )

    candidate = resolve_browser_cookie_candidate("auto")

    assert candidate is not None
    assert candidate.browser == "firefox"
    assert candidate.source == "default_browser"


def test_apply_youtube_cookie_config_writes_browser_reference_only(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
api:
  base_url: https://llm.test/v1
source:
  resolution: "720"
""".strip(),
        encoding="utf-8",
    )

    result = apply_youtube_cookie_config(config_path, browser="firefox", profile="default-release")
    data = load_mapping(config_path)

    assert result["updated"] is True
    assert result["candidate"]["browser"] == "firefox"
    assert data["api"]["base_url"] == "https://llm.test/v1"
    assert data["source"]["resolution"] == "720"
    assert data["source"]["cookies_from_browser"] == "firefox"
    assert data["source"]["cookies_browser_profile"] == "default-release"
    assert data["youtube"]["cookies_from_browser"] == "firefox"
