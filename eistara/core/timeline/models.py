from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eistara.core.subtitle import SubtitleEvent


@dataclass(frozen=True, slots=True)
class TimelineInput:
    segment_id: str
    source_start_sec: float
    source_end_sec: float
    target_text: str
    source_text: str = ""
    speaker: str = "SPEAKER_00"
    audio_path: Path | None = None
    audio_duration_sec: float | None = None

    @property
    def source_duration_sec(self) -> float:
        return max(0.0, float(self.source_end_sec) - float(self.source_start_sec))

    @property
    def has_dub_content(self) -> bool:
        return bool(self.target_text.strip()) and self.audio_duration_sec is not None and self.audio_duration_sec > 0


@dataclass(frozen=True, slots=True)
class DubTimelineSegment:
    segment_id: str
    source_start_sec: float
    source_end_sec: float
    dub_start_sec: float
    dub_end_sec: float
    target_text: str
    source_text: str = ""
    speaker: str = "SPEAKER_00"
    audio_path: Path | None = None
    audio_duration_sec: float = 0.0

    @property
    def source_duration_sec(self) -> float:
        return max(0.0, float(self.source_end_sec) - float(self.source_start_sec))

    @property
    def dub_duration_sec(self) -> float:
        return max(0.0, float(self.dub_end_sec) - float(self.dub_start_sec))

    def to_subtitle_event(self) -> SubtitleEvent:
        return SubtitleEvent(self.dub_start_sec, self.dub_end_sec, (self.target_text,))


@dataclass(frozen=True, slots=True)
class DubTimeline:
    segments: tuple[DubTimelineSegment, ...]
    warnings: tuple[str, ...] = ()
    tail_pad_sec: float = 0.0

    @property
    def duration_sec(self) -> float:
        if not self.segments:
            return 0.0
        return max(segment.dub_end_sec for segment in self.segments) + max(0.0, float(self.tail_pad_sec))

    def subtitle_events(self) -> list[SubtitleEvent]:
        return [segment.to_subtitle_event() for segment in self.segments if segment.target_text.strip()]
