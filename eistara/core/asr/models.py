from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AsrSettings:
    language: str | None = None
    model: str | None = None
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AsrRequest:
    audio_path: Path
    language: str | None = None
    prompt: str = ""


@dataclass(frozen=True, slots=True)
class AsrSegment:
    id: int
    start_sec: float
    end_sec: float
    text: str
    speaker: str = "SPEAKER_00"
    words: tuple[dict[str, Any], ...] = ()

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "text": self.text,
            "speaker": self.speaker,
            "duration_sec": self.duration_sec,
            "words": list(self.words),
        }


@dataclass(frozen=True, slots=True)
class AsrResult:
    segments: tuple[AsrSegment, ...]
    language: str | None = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "segments": [segment.to_dict() for segment in self.segments],
            "warnings": list(self.warnings),
        }
