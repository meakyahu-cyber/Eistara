from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any, Mapping, Protocol
import os
import re
import shutil
import subprocess
import sys
import time

if sys.platform.startswith("win"):
    import winreg

from eistara.core.source import (
    SourceProviderError,
    SourceRequest,
    SourceResult,
    SourceSettings,
    allowed_video_formats,
)
from eistara.config.youtube_cookies import resolve_browser_cookie_candidate


GENERATED_VIDEO_NAMES = {
    "output.mp4",
    "output_dub.mp4",
    "output_sub.mp4",
    "output_dub.webm",
    "output_sub.webm",
}


class _FilteredYtdlpStream:
    """Suppress native yt-dlp progress noise while preserving real errors."""

    def __init__(self, stream):
        self.stream = stream
        self._buffer = ""

    def write(self, text):
        if not text:
            return 0
        self._buffer += str(text).replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._write_line(line)
        return len(text)

    def flush(self):
        if self._buffer:
            self._write_line(self._buffer)
            self._buffer = ""
        try:
            self.stream.flush()
        except Exception:
            pass

    def isatty(self):
        return bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")

    def _write_line(self, line):
        stripped = line.strip()
        if not stripped:
            return
        plain = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stripped)
        if plain.startswith("[download]"):
            return
        self.stream.write(line + "\n")


class _YtdlpLogger:
    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        print(msg)

    def error(self, msg):
        print(msg)


class ProcessRunner(Protocol):
    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        """Run a process and return a completed process."""


class YtDlpProcessRunner:
    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        import subprocess

        return subprocess.run(args, capture_output=True, text=True, check=False)


@dataclass(slots=True)
class YtDlpSourceProvider:
    runner: ProcessRunner | None = None
    executable: str = "yt-dlp"
    name: str = "yt-dlp"

    def __post_init__(self) -> None:
        if self.runner is None:
            self.runner = YtDlpProcessRunner()

    def acquire(self, request: SourceRequest, settings: SourceSettings) -> SourceResult:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        if _config_bool(settings.provider_config.get("use_python_api"), True):
            return self._acquire_python_api(request, settings)
        output_template = _download_output_template(request.output_dir)
        args = build_ytdlp_args(
            request.source,
            output_template,
            request.resolution,
            self.executable,
            extra_args=_extra_args(settings),
        )
        result = self.runner.run(args)
        if result.returncode != 0:
            raise SourceProviderError((result.stderr or result.stdout or "yt-dlp failed").strip())
        _rename_sanitized_files(request.output_dir)
        source_video = _find_downloaded_video(request.output_dir, allowed_video_formats(settings))
        if source_video is None:
            raise SourceProviderError(f"yt-dlp completed but no unique video file was found in {request.output_dir}")
        return SourceResult(
            source_video=source_video,
            source_type="url",
            metadata={"download_command": list(args)},
        )

    def _acquire_python_api(self, request: SourceRequest, settings: SourceSettings) -> SourceResult:
        YoutubeDL = _youtube_dl_class(settings)
        output_template = _download_output_template(request.output_dir)
        options = build_ytdlp_options(request.source, output_template, request.resolution, settings)
        try:
            _download_progress_hook._last_report = 0
            with redirect_stdout(_FilteredYtdlpStream(sys.stdout)), redirect_stderr(_FilteredYtdlpStream(sys.stderr)):
                with YoutubeDL(options) as ydl:
                    ydl.download([request.source])
        except Exception as exc:
            raise SourceProviderError(str(exc)) from exc
        _rename_sanitized_files(request.output_dir)
        source_video = _find_downloaded_video(request.output_dir, allowed_video_formats(settings))
        if source_video is None:
            raise SourceProviderError(f"yt-dlp completed but no unique video file was found in {request.output_dir}")
        return SourceResult(
            source_video=source_video,
            source_type="url",
            metadata={"download_options": _safe_metadata_options(options)},
        )


def build_ytdlp_args(
    source: str,
    output_template: str,
    resolution: str = "",
    executable: str = "yt-dlp",
    extra_args: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    args = [executable, "-o", output_template]
    if resolution == "best":
        args.extend(["-f", "bestvideo+bestaudio/best"])
    elif resolution:
        args.extend(["-f", f"bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]"])
    else:
        args.extend(["-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]"])
    args.extend(str(item) for item in (extra_args or ()))
    args.append(source)
    return tuple(args)


def _extra_args(settings: SourceSettings) -> list[str]:
    raw = settings.provider_config.get("yt_dlp_extra_args") or settings.provider_config.get("extra_args") or []
    if isinstance(raw, str):
        return [item for item in raw.split() if item]
    if isinstance(raw, (list, tuple)):
        return [str(item) for item in raw]
    return []


def build_ytdlp_options(source: str, output_template: str, resolution: str, settings: SourceSettings) -> dict:
    config = settings.provider_config
    options = {
        "format": "bestvideo+bestaudio/best" if resolution == "best" else f"bestvideo[height<={resolution or '1080'}]+bestaudio/best[height<={resolution or '1080'}]",
        "outtmpl": output_template,
        "noplaylist": True,
        "logger": _YtdlpLogger(),
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "progress_delta": 5,
        "consoletitle": False,
        "color": {"stdout": "auto", "stderr": "auto"},
        "progress_hooks": [_download_progress_hook],
    }
    if _config_bool(config.get("write_thumbnail"), False):
        options["writethumbnail"] = True
        options["postprocessors"] = [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}]
    apply_common_ytdlp_options(options, config)
    return options


def apply_common_ytdlp_options(options: dict[str, Any], provider_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = provider_config or {}
    options["socket_timeout"] = float(config.get("socket_timeout") or 20)
    options["retries"] = int(config.get("retries") or 3)
    options["fragment_retries"] = int(config.get("retries") or 3)

    js_runtimes = _build_js_runtimes()
    if js_runtimes:
        options["js_runtimes"] = js_runtimes
    proxy = _build_proxy(str(config.get("proxy") or "").strip())
    if proxy:
        options["proxy"] = proxy
    cookies_path = str(config.get("cookies_path") or "").strip()
    if cookies_path and Path(cookies_path).exists():
        options["cookiefile"] = cookies_path
    else:
        cookies_browser = str(config.get("cookies_from_browser") or "").strip()
        if cookies_browser:
            cookies_profile = str(config.get("cookies_browser_profile") or "").strip() or None
            resolved = resolve_browser_cookie_candidate(cookies_browser, cookies_profile or "")
            if resolved:
                cookies_browser = resolved.browser
                cookies_profile = resolved.profile or cookies_profile
            options["cookiesfrombrowser"] = (cookies_browser, cookies_profile, None, None)
    return options


def sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    filename = filename.strip(". ")
    return filename if filename else "video"


def _download_output_template(output_dir: Path) -> str:
    return str(output_dir / "%(title)s.%(ext)s")


def _youtube_dl_class(settings: SourceSettings):
    if _config_bool(settings.provider_config.get("auto_update_ytdlp"), False):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "yt-dlp"])
            if "yt_dlp" in sys.modules:
                del sys.modules["yt_dlp"]
        except subprocess.CalledProcessError as exc:
            print(f"Warning: Failed to update yt-dlp: {exc}")
    try:
        from yt_dlp import YoutubeDL
    except Exception as exc:
        raise SourceProviderError("yt-dlp Python package is not available") from exc
    return YoutubeDL


def _download_progress_hook(status):
    if status.get("status") == "downloading":
        now = time.time()
        last_report = getattr(_download_progress_hook, "_last_report", 0)
        if now - last_report < 5:
            return
        _download_progress_hook._last_report = now
        filename = os.path.basename(status.get("filename") or "")
        percent = status.get("_percent_str", "").strip()
        speed = status.get("_speed_str", "").strip()
        eta = status.get("_eta_str", "").strip()
        if percent:
            print(f"Downloading {filename}: {percent} at {speed}, ETA {eta}")
    elif status.get("status") == "finished":
        filename = os.path.basename(status.get("filename") or "")
        print(f"Downloaded {filename}; processing...")


def _rename_sanitized_files(output_dir: Path) -> None:
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        new_name = sanitize_filename(path.stem) + path.suffix
        if new_name != path.name:
            path.rename(path.with_name(new_name))


def _config_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _build_js_runtimes() -> dict | None:
    for runtime in ("node", "deno", "bun", "quickjs"):
        if shutil.which(runtime):
            return {runtime: {}}
    return None


def _normalize_proxy(proxy: str | None) -> str | None:
    if not proxy:
        return None
    proxy = proxy.strip()
    if not proxy:
        return None
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    return proxy


def _proxy_from_windows_settings() -> str | None:
    if not sys.platform.startswith("win"):
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
            proxy_enabled = winreg.QueryValueEx(key, "ProxyEnable")[0]
            if not proxy_enabled:
                return None
            proxy_server = winreg.QueryValueEx(key, "ProxyServer")[0]
    except OSError:
        return None

    if "=" not in proxy_server:
        return _normalize_proxy(proxy_server)

    proxy_parts = {}
    for item in proxy_server.split(";"):
        if "=" not in item:
            continue
        scheme, value = item.split("=", 1)
        proxy_parts[scheme.strip().lower()] = value.strip()
    return _normalize_proxy(proxy_parts.get("https") or proxy_parts.get("http") or proxy_parts.get("socks"))


def _build_proxy(explicit_proxy: str | None = None) -> str | None:
    explicit = _normalize_proxy(explicit_proxy)
    if explicit:
        return explicit
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        proxy = _normalize_proxy(os.environ.get(key))
        if proxy:
            return proxy
    return _proxy_from_windows_settings()


def _safe_metadata_options(options: dict) -> dict:
    safe = {}
    for key, value in options.items():
        if key in {"logger", "progress_hooks"}:
            continue
        if key == "cookiesfrombrowser":
            browser, profile, _, _ = value
            safe[key] = [browser, profile, None, None]
        else:
            safe[key] = _json_safe(value)
    return safe


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _find_downloaded_video(output_dir: Path, allowed_formats: set[str]) -> Path | None:
    candidates = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and path.suffix.lower().lstrip(".") in allowed_formats
        and path.name not in GENERATED_VIDEO_NAMES
        and not path.name.startswith("output_")
    ]
    if len(candidates) > 1:
        raise SourceProviderError(f"Number of videos found {len(candidates)} is not unique. Please check.")
    return candidates[0] if candidates else None
