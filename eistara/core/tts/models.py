from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TtsSettings:
    method: str = "indextts"
    cache_version: int = 2
    max_retries: int = 3
    service_backoff_base_sec: float = 2.0
    provider_config: dict[str, Any] = field(default_factory=dict)
    audio_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TtsRequest:
    text: str
    output_path: Path
    segment_id: str | int | None = None
    voice: str | None = None
    speaker: str = "SPEAKER_00"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TtsResult:
    output_path: Path
    cached: bool = False
    duration_sec: float | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
