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
    max_batch_lines: int = 30
    max_batch_chars: int = 3000
    use_summary: bool = True
    summary_length: int = 8000
    enforce_latin: bool = True
    pacing_budget_enabled: bool = True
    estimated_tts_chars_per_sec: float = 4.4
    estimated_zh_chars_per_en_word: float = 1.72
    min_zh_chars_per_en_word: float = 1.67
    natural_min_zh_chars_per_en_word: float = 1.70
    natural_max_zh_chars_per_en_word: float = 1.72
    soft_pressure_max_zh_chars_per_en_word: float = 1.75
    hard_pressure_max_zh_chars_per_en_word: float = 1.73
    critical_pressure_max_zh_chars_per_en_word: float = 1.70
    target_pacing_pressure: float = 1.05
    soft_pacing_pressure: float = 1.10
    hard_pacing_pressure: float = 1.18
    critical_pacing_pressure: float = 1.25
    min_pacing_source_sec: float = 20.0
    raw_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TranslationResult:
    translations: dict[int, str]
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class TranslationPacingBudget:
    enabled: bool
    source_duration_sec: float
    english_words: int
    english_words_per_sec: float
    estimated_zh_chars_per_en_word: float
    min_zh_chars_per_en_word: float
    natural_min_zh_chars_per_en_word: float
    natural_max_zh_chars_per_en_word: float
    soft_pressure_max_zh_chars_per_en_word: float
    hard_pressure_max_zh_chars_per_en_word: float
    critical_pressure_max_zh_chars_per_en_word: float
    target_zh_chars_per_en_word: float
    max_zh_chars_per_en_word: float
    estimated_tts_chars_per_sec: float
    predicted_pressure: float
    pressure_at_min_zh_ratio: float
    target_pressure: float
    soft_pressure: float
    ideal_min_zh_chars: int
    ideal_max_zh_chars: int
    min_zh_chars: int
    target_zh_chars: int
    hard_zh_chars: int
    level: str
    reason: str = ""

    def to_prompt_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "level": self.level,
            "source_duration_sec": round(self.source_duration_sec, 3),
            "english_words": self.english_words,
            "english_words_per_sec": round(self.english_words_per_sec, 3),
            "estimated_zh_chars_per_en_word": round(self.estimated_zh_chars_per_en_word, 3),
            "min_zh_chars_per_en_word": round(self.min_zh_chars_per_en_word, 3),
            "natural_min_zh_chars_per_en_word": round(self.natural_min_zh_chars_per_en_word, 3),
            "natural_max_zh_chars_per_en_word": round(self.natural_max_zh_chars_per_en_word, 3),
            "soft_pressure_max_zh_chars_per_en_word": round(self.soft_pressure_max_zh_chars_per_en_word, 3),
            "hard_pressure_max_zh_chars_per_en_word": round(self.hard_pressure_max_zh_chars_per_en_word, 3),
            "critical_pressure_max_zh_chars_per_en_word": round(self.critical_pressure_max_zh_chars_per_en_word, 3),
            "target_zh_chars_per_en_word": round(self.target_zh_chars_per_en_word, 3),
            "max_zh_chars_per_en_word": round(self.max_zh_chars_per_en_word, 3),
            "estimated_tts_chars_per_sec": round(self.estimated_tts_chars_per_sec, 3),
            "predicted_pressure": round(self.predicted_pressure, 3),
            "pressure_at_min_zh_ratio": round(self.pressure_at_min_zh_ratio, 3),
            "target_pressure": round(self.target_pressure, 3),
            "soft_pressure": round(self.soft_pressure, 3),
            "ideal_min_zh_chars": self.ideal_min_zh_chars,
            "ideal_max_zh_chars": self.ideal_max_zh_chars,
            "min_zh_chars": self.min_zh_chars,
            "target_zh_chars": self.target_zh_chars,
            "hard_zh_chars": self.hard_zh_chars,
            "reason": self.reason,
        }
