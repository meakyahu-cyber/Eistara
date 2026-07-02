from __future__ import annotations

from pathlib import Path

from eistara.core.media import AudioStreamInfo, MediaInfo, source_duration_sec
from eistara.core.pipeline import StageContext
from eistara.core.jobs.models import StageName


class FakeProbe:
    def __init__(self, durations: dict[str, float | None], *, fail_paths: set[str] | None = None) -> None:
        self.durations = durations
        self.fail_paths = fail_paths or set()
        self.calls: list[str] = []

    def probe(self, path: str) -> MediaInfo:
        self.calls.append(path)
        if path in self.fail_paths:
            raise RuntimeError("probe failed")
        duration = self.durations.get(path)
        if duration == 0:
            return MediaInfo(Path(path), duration_sec=0.0)
        if duration is None:
            return MediaInfo(Path(path), audio=AudioStreamInfo(duration_sec=3.25))
        return MediaInfo(Path(path), duration_sec=duration)


def test_source_duration_sec_uses_first_available_source_audio_duration() -> None:
    context = StageContext(
        "job",
        Path("."),
        {
            "high_quality_audio": "hq.wav",
            "raw_audio": "raw.wav",
        },
        StageName.TTS,
        1,
    )

    duration = source_duration_sec(context, FakeProbe({"hq.wav": 4.5, "raw.wav": 9.0}))

    assert duration == 4.5


def test_source_duration_sec_skips_probe_errors_and_falls_back_to_audio_stream_duration() -> None:
    context = StageContext(
        "job",
        Path("."),
        {"high_quality_audio": "bad.wav"},
        StageName.TTS,
        1,
        artifacts={"raw_audio": "raw.wav"},
    )
    probe = FakeProbe({"raw.wav": None}, fail_paths={"bad.wav"})

    duration = source_duration_sec(context, probe)

    assert duration == 3.25
    assert probe.calls == ["bad.wav", "raw.wav"]
