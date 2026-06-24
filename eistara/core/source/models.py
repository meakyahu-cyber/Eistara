from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_ALLOWED_VIDEO_FORMATS = ("mp4", "mov", "avi", "mkv", "flv", "wmv", "webm")


@dataclass(frozen=True, slots=True)
class SourceSettings:
    output_filename: str = "source_video.mp4"
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceRequest:
    source: str
    source_type: str
    output_dir: Path
    title: str = ""
    resolution: str = ""

    @property
    def output_path(self) -> Path:
        return self.output_dir / "source_video.mp4"


@dataclass(frozen=True, slots=True)
class SourceResult:
    source_video: Path
    source_type: str
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


def allowed_video_formats(settings: SourceSettings) -> set[str]:
    raw = settings.provider_config.get("allowed_video_formats") or DEFAULT_ALLOWED_VIDEO_FORMATS
    if isinstance(raw, str):
        return {item.strip().lower().lstrip(".") for item in raw.split(",") if item.strip()}
    if isinstance(raw, (list, tuple, set)):
        return {str(item).strip().lower().lstrip(".") for item in raw if str(item).strip()}
    return set(DEFAULT_ALLOWED_VIDEO_FORMATS)
