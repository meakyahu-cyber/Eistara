from __future__ import annotations

from typing import Protocol

from .models import MediaInfo
from .plans import AudioExtractPlan, ComposeVideoPlan, MediaCommandResult


class MediaProbe(Protocol):
    def probe(self, path: str) -> MediaInfo:
        """Read media metadata."""


class MediaProvider(Protocol):
    name: str

    def probe(self, path: str) -> MediaInfo:
        """Read media metadata."""

    def extract_audio(self, plan: AudioExtractPlan) -> MediaCommandResult:
        """Extract source audio according to a plan."""

    def compose_video(self, plan: ComposeVideoPlan) -> MediaCommandResult:
        """Compose final video according to a plan."""
