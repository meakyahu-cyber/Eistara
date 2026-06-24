from __future__ import annotations

from pathlib import Path

from eistara.core.media import AudioStreamInfo, MediaInfo
from eistara.core.timeline import TimelinePreparationService


class FakeProbe:
    def __init__(self, duration: float = 1.25):
        self.duration = duration
        self.calls = []

    def probe(self, path: str) -> MediaInfo:
        self.calls.append(path)
        return MediaInfo(path=Path(path), duration_sec=self.duration, audio=AudioStreamInfo(duration_sec=self.duration))


def test_timeline_preparation_builds_segments_from_tts_segments() -> None:
    service = TimelinePreparationService(FakeProbe(1.5))

    segments, warnings = service.build_segments(
        [
            {
                "id": "1",
                "start": 0,
                "end": 2,
                "source": "source line",
                "text": "hello",
                "output_path": "audio/1.wav",
            }
        ]
    )

    assert warnings == []
    assert segments[0]["id"] == "1"
    assert segments[0]["source"] == "source line"
    assert segments[0]["target"] == "hello"
    assert segments[0]["audio_path"] == "audio/1.wav"
    assert segments[0]["audio_duration_sec"] == 1.5


def test_timeline_preparation_uses_tts_outputs_mapping() -> None:
    service = TimelinePreparationService(FakeProbe(0.8))

    segments, warnings = service.build_segments(
        [{"id": "a", "start": 0, "end": 1, "target": "line"}],
        tts_outputs={"a": "a.wav"},
    )

    assert warnings == []
    assert segments[0]["audio_path"] == "a.wav"
    assert segments[0]["audio_duration_sec"] == 0.8


def test_timeline_preparation_uses_tts_duration_mapping_before_probe() -> None:
    service = TimelinePreparationService(FakeProbe(9.0))

    segments, warnings = service.build_segments(
        [{"id": "a", "start": 0, "end": 1, "target": "line"}],
        tts_outputs={"a": "a.wav"},
        tts_durations={"a": 1.4},
    )

    assert warnings == []
    assert segments[0]["audio_duration_sec"] == 1.4


def test_timeline_preparation_warns_missing_duration() -> None:
    segments, warnings = TimelinePreparationService().build_segments([{"id": "1", "text": "hello"}])

    assert segments[0]["audio_duration_sec"] is None
    assert warnings == ["1: missing audio duration"]


def test_timeline_preparation_writes_json(tmp_path: Path) -> None:
    path, segments, warnings = TimelinePreparationService(FakeProbe(1)).write_segments(
        [{"id": "1", "text": "hello", "output_path": "a.wav"}],
        tmp_path / "dub_segments.json",
    )

    assert path.exists()
    assert segments[0]["audio_duration_sec"] == 1
    assert warnings == []
