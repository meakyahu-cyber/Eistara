from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class TranslationItem:
    id: int
    source: str
    start: str = ""
    end: str = ""
    duration_sec: float | None = None
    speaker: str = "SPEAKER_00"

    def to_prompt_dict(self) -> dict:
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "duration_sec": self.duration_sec,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class Terminology:
    theme: str = ""
    terms: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class TranslationSettings:
    source_language: str = "source language"
    target_language: str = "Simplified Chinese"
    max_batch_lines: int = 20
    max_batch_chars: int = 3000
    use_summary: bool = True
    summary_length: int = 8000
    enforce_latin: bool = True
    localization_chars_per_sec: float = 4.2
    localization_spoken_cost_per_sec: float = 3.6
    localization_max_audio_speed: float = 1.10
    localization_seam_gap_sec: float = 0.12
    localization_max_window_gap_sec: float = 6.0
    raw_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranslationResult:
    translations: dict[int, str]
    warnings: list[str] = field(default_factory=list)
    localization_reports: list[dict[str, Any]] = field(default_factory=list)
