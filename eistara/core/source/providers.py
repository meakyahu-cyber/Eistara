from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Protocol

from .models import SourceRequest, SourceResult, SourceSettings, allowed_video_formats


class SourceProviderError(RuntimeError):
    """Raised when source acquisition fails."""


class SourceProvider(Protocol):
    name: str

    def acquire(self, request: SourceRequest, settings: SourceSettings) -> SourceResult:
        """Acquire a source video and return a local path."""


class LocalFileSourceProvider:
    name = "local-file"

    def acquire(self, request: SourceRequest, settings: SourceSettings) -> SourceResult:
        source_path = Path(request.source)
        if not source_path.is_file():
            raise SourceProviderError(f"Source file not found: {source_path}")
        allowed = allowed_video_formats(settings)
        ext = source_path.suffix.lower().lstrip(".")
        if ext not in allowed:
            raise SourceProviderError(f"Only video source files are supported here, got: {source_path.suffix}")
        output_dir = Path(request.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / _target_filename(source_path, settings)
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)
        return SourceResult(source_video=target, source_type="file")


def _target_filename(source_path: Path, settings: SourceSettings) -> str:
    configured = str(settings.output_filename or "").strip()
    if configured and configured != "source_video.mp4":
        return configured
    return _sanitize_filename(source_path.stem) + source_path.suffix.lower()


def _sanitize_filename(filename: str) -> str:
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    filename = filename.strip(". ")
    return filename if filename else "video"
