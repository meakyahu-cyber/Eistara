from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import AsrRequest, AsrResult, AsrSegment, AsrSettings


class AsrProviderError(RuntimeError):
    """Raised when ASR cannot complete."""


class AsrProvider(Protocol):
    name: str

    def transcribe(self, request: AsrRequest, settings: AsrSettings) -> AsrResult:
        """Return transcription segments for an audio file."""


class VocalSeparationProvider(Protocol):
    name: str

    def separate(self, source_audio: str | Path, output_dir: str | Path, *, segment_minutes: float = 30.0) -> tuple[Path, Path]:
        """Return vocal/background audio paths under the job output directory."""

    def normalize(self, audio_path: str | Path, output_path: str | Path, *, format: str = "mp3") -> Path:
        """Normalize vocal audio in place or to a target path."""


class ScriptedAsrProvider:
    name = "scripted"

    def __init__(self, segments: list[AsrSegment] | None = None, language: str | None = "en"):
        self.segments = tuple(segments or [])
        self.language = language
        self.calls: list[AsrRequest] = []

    def transcribe(self, request: AsrRequest, settings: AsrSettings) -> AsrResult:
        self.calls.append(request)
        return AsrResult(self.segments, language=request.language or settings.language or self.language)
