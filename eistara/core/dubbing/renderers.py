from __future__ import annotations

from typing import Protocol

from eistara.core.media import ComposeVideoPlan, MediaCommandResult

from .models import AudioMixPlan


class DubbingRenderer(Protocol):
    name: str

    def render_audio_mix(self, plan: AudioMixPlan) -> MediaCommandResult:
        """Render an audio mix plan into plan.output_audio."""

    def render_video(self, plan: ComposeVideoPlan) -> MediaCommandResult:
        """Render a final video from a compose plan."""
