from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.media import ComposeVideoPlan


@dataclass(frozen=True, slots=True)
class AudioClipPlacement:
    segment_id: str
    audio_path: Path
    start_sec: float
    end_sec: float
    gain_db: float = 0.0
    speed: float = 1.0

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))

    @property
    def effective_duration_sec(self) -> float:
        return self.duration_sec / max(1.0, float(self.speed))

    @property
    def effective_end_sec(self) -> float:
        return float(self.start_sec) + self.effective_duration_sec

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "audio_path": str(self.audio_path),
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "duration_sec": self.duration_sec,
            "speed": self.speed,
            "effective_duration_sec": self.effective_duration_sec,
            "effective_end_sec": self.effective_end_sec,
            "gain_db": self.gain_db,
        }


@dataclass(frozen=True, slots=True)
class AudioMixPlan:
    clips: tuple[AudioClipPlacement, ...]
    output_audio: Path
    duration_sec: float
    sample_rate_hz: int = 24000
    channels: int = 2
    bitrate: str = "192k"
    pre_speed_duration_sec: float | None = None
    global_audio_speed: float = 1.0
    background_audio: Path | None = None
    background_gain_db: float = -18.0
    clip_lowpass_hz: int = 6800
    clip_peak_normalize_dbfs: float | None = -3.0
    clip_fade_in_ms: int = 5
    clip_fade_out_ms: int = 220
    clip_tail_pad_ms: int = 220
    clip_tail_pad_counts_in_timeline: bool = False
    clip_tail_cleanup: bool = True
    clip_tail_cleanup_ms: int = 420
    clip_tail_cleanup_lowpass_hz: int = 3600
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_audio": str(self.output_audio),
            "duration_sec": self.duration_sec,
            "sample_rate_hz": self.sample_rate_hz,
            "channels": self.channels,
            "bitrate": self.bitrate,
            "pre_speed_duration_sec": self.pre_speed_duration_sec,
            "global_audio_speed": self.global_audio_speed,
            "background_audio": str(self.background_audio) if self.background_audio else None,
            "background_gain_db": self.background_gain_db,
            "clip_lowpass_hz": self.clip_lowpass_hz,
            "clip_peak_normalize_dbfs": self.clip_peak_normalize_dbfs,
            "clip_fade_in_ms": self.clip_fade_in_ms,
            "clip_fade_out_ms": self.clip_fade_out_ms,
            "clip_tail_pad_ms": self.clip_tail_pad_ms,
            "clip_tail_pad_counts_in_timeline": self.clip_tail_pad_counts_in_timeline,
            "clip_tail_cleanup": self.clip_tail_cleanup,
            "clip_tail_cleanup_ms": self.clip_tail_cleanup_ms,
            "clip_tail_cleanup_lowpass_hz": self.clip_tail_cleanup_lowpass_hz,
            "clips": [clip.to_dict() for clip in self.clips],
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class DubbingRenderPlan:
    audio_mix: AudioMixPlan
    compose_video: ComposeVideoPlan
    dub_subtitle: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio_mix": self.audio_mix.to_dict(),
            "compose_video": {
                "source_video": str(self.compose_video.source_video),
                "dub_audio": str(self.compose_video.dub_audio),
                "output_video": str(self.compose_video.output_video),
                "subtitle_path": str(self.compose_video.subtitle_path) if self.compose_video.subtitle_path else None,
                "background_audio": str(self.compose_video.background_audio) if self.compose_video.background_audio else None,
                "copy_video": self.compose_video.copy_video,
            },
            "dub_subtitle": str(self.dub_subtitle) if self.dub_subtitle else None,
        }
