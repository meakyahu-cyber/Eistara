from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from eistara.adapters.source.ytdlp import (
    YtDlpSourceProvider,
    build_ytdlp_args,
    build_ytdlp_options,
)
from eistara.core.source import SourceProviderError, SourceRequest, SourceSettings


@dataclass(slots=True)
class FakeRunner:
    result: CompletedProcess[str]
    calls: list[tuple[str, ...]]
    output_file: Path | None = None

    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        self.calls.append(args)
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_bytes(b"video")
        return self.result


def test_build_ytdlp_args_uses_resolution_filter() -> None:
    args = build_ytdlp_args("https://example.com/v", "out.%(ext)s", "720", "yt-dlp.exe")

    assert args[0] == "yt-dlp.exe"
    assert "height<=720" in args[args.index("-f") + 1]


def test_build_ytdlp_args_accepts_extra_args() -> None:
    args = build_ytdlp_args(
        "https://example.com/v",
        "out.%(ext)s",
        executable="yt-dlp.exe",
        extra_args=["--cookies-from-browser", "chrome"],
    )

    assert args[-3:] == ("--cookies-from-browser", "chrome", "https://example.com/v")


def test_ytdlp_source_provider_returns_downloaded_file(tmp_path: Path) -> None:
    runner = FakeRunner(CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""), [], tmp_path / "output" / "source_video.webm")
    provider = YtDlpSourceProvider(runner=runner, executable="yt-dlp.exe")

    result = provider.acquire(
        SourceRequest("https://example.com/v", "url", tmp_path / "output", resolution="720"),
        SourceSettings(provider_config={"use_python_api": False}),
    )

    assert result.source_video == tmp_path / "output" / "source_video.webm"
    assert runner.calls[0][0] == "yt-dlp.exe"
    assert runner.calls[0][2].endswith("%(title)s.%(ext)s")
    assert result.metadata["download_command"] == list(runner.calls[0])


def test_ytdlp_source_provider_ignores_generated_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "output_dub.mp4").write_bytes(b"rendered")
    runner = FakeRunner(CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""), [], output_dir / "Video Title.webm")
    provider = YtDlpSourceProvider(runner=runner, executable="yt-dlp.exe")

    result = provider.acquire(
        SourceRequest("https://example.com/v", "url", output_dir, resolution="720"),
        SourceSettings(provider_config={"use_python_api": False}),
    )

    assert result.source_video == output_dir / "Video Title.webm"


def test_ytdlp_source_provider_raises_on_failure(tmp_path: Path) -> None:
    provider = YtDlpSourceProvider(runner=FakeRunner(CompletedProcess(args=[], returncode=1, stdout="", stderr="bad"), []))

    try:
        provider.acquire(SourceRequest("https://example.com/v", "url", tmp_path / "output"), SourceSettings(provider_config={"use_python_api": False}))
    except SourceProviderError as exc:
        assert "bad" in str(exc)
    else:
        raise AssertionError("expected SourceProviderError")


def test_build_ytdlp_options_uses_browser_cookies_and_runtime() -> None:
    options = build_ytdlp_options(
        "https://example.com/v",
        "out.%(ext)s",
        "720",
        SourceSettings(provider_config={"cookies_from_browser": "firefox", "cookies_browser_profile": "default"}),
    )

    assert options["cookiesfrombrowser"] == ("firefox", "default", None, None)
    assert "height<=720" in options["format"]
    assert options["progress_delta"] == 5


def test_build_ytdlp_options_accepts_explicit_browser_alias() -> None:
    options = build_ytdlp_options(
        "https://example.com/v",
        "out.%(ext)s",
        "720",
        SourceSettings(provider_config={"cookies_from_browser": "microsoft edge", "cookies_browser_profile": "Default"}),
    )

    assert options["cookiesfrombrowser"] == ("edge", "Default", None, None)


def test_build_ytdlp_options_uses_v1_env_proxy_detection(monkeypatch) -> None:
    for key in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "127.0.0.1:7890")

    options = build_ytdlp_options("https://example.com/v", "out.%(ext)s", "720", SourceSettings())

    assert options["proxy"] == "http://127.0.0.1:7890"


def test_build_ytdlp_options_prefers_explicit_proxy_over_env(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")

    options = build_ytdlp_options(
        "https://example.com/v",
        "out.%(ext)s",
        "720",
        SourceSettings(provider_config={"proxy": "http://127.0.0.1:8899"}),
    )

    assert options["proxy"] == "http://127.0.0.1:8899"


def test_build_ytdlp_options_enables_thumbnail_postprocessor() -> None:
    options = build_ytdlp_options(
        "https://example.com/v",
        "out.%(ext)s",
        "720",
        SourceSettings(provider_config={"write_thumbnail": True}),
    )

    assert options["writethumbnail"] is True
    assert options["postprocessors"] == [{"key": "FFmpegThumbnailsConvertor", "format": "jpg"}]


