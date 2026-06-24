from __future__ import annotations

from dataclasses import dataclass
from subprocess import CompletedProcess

from eistara.adapters.media import FfmpegMediaProvider
from eistara.core.media import MediaProbeError, build_audio_extract_plan


@dataclass(slots=True)
class FakeRunner:
    result: CompletedProcess[str]
    calls: list[tuple[str, ...]]

    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        self.calls.append(args)
        return self.result


def test_ffmpeg_provider_probe_parses_json() -> None:
    runner = FakeRunner(
        CompletedProcess(
            args=[],
            returncode=0,
            stdout='{"format":{"duration":"1.0"},"streams":[{"codec_type":"audio","codec_name":"aac"}]}',
            stderr="",
        ),
        [],
    )
    provider = FfmpegMediaProvider(runner=runner, ffprobe_path="ffprobe.exe")

    info = provider.probe("input.mp4")

    assert info.duration_sec == 1.0
    assert info.has_audio
    assert runner.calls[0][0] == "ffprobe.exe"


def test_ffmpeg_provider_probe_raises_on_failure() -> None:
    provider = FfmpegMediaProvider(
        runner=FakeRunner(CompletedProcess(args=[], returncode=1, stdout="", stderr="missing file"), [])
    )

    try:
        provider.probe("missing.mp4")
    except MediaProbeError as exc:
        assert "missing file" in str(exc)
    else:
        raise AssertionError("expected MediaProbeError")


def test_ffmpeg_provider_extract_audio_returns_command_result(tmp_path) -> None:
    runner = FakeRunner(CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""), [])
    provider = FfmpegMediaProvider(runner=runner, ffmpeg_path="ffmpeg.exe")

    result = provider.extract_audio(build_audio_extract_plan(tmp_path / "source.mp4", tmp_path / "raw.wav"))

    assert result.ok
    assert result.command[0] == "ffmpeg.exe"
    assert runner.calls[0] == result.command
