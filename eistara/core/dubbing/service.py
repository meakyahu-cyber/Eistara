from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eistara.core.media import ComposeVideoPlan, build_compose_video_plan
from eistara.core.timeline import DubTimeline

from .models import AudioClipPlacement, AudioMixPlan, DubbingRenderPlan


@dataclass(frozen=True, slots=True)
class DubbingRenderService:
    output_audio_name: str = "dub.mp3"
    output_video_name: str = "output_dub.mp4"
    sample_rate_hz: int = 24000
    channels: int = 2
    bitrate: str = "192k"
    clip_gain_db: float = 0.0
    background_gain_db: float = -18.0
    clip_lowpass_hz: int = 6800
    clip_peak_normalize_dbfs: float | None = -3.0
    clip_fade_in_ms: int = 5
    clip_fade_out_ms: int = 220
    clip_tail_pad_ms: int = 220
    clip_tail_cleanup: bool = True
    clip_tail_cleanup_ms: int = 420
    clip_tail_cleanup_lowpass_hz: int = 3600
    publish_global_audio_speed: bool = True
    publish_target_video_speed_min: float = 0.88
    publish_max_audio_speed: float = 1.22
    publish_short_video_speed_max: float = 1.05
    publish_short_video_speed_hard_max: float = 1.08

    def audio_mix_plan(
        self,
        timeline: DubTimeline,
        output_dir: str | Path,
        *,
        background_audio: str | Path | None = None,
        output_duration_sec: float | None = None,
        pre_speed_duration_sec: float | None = None,
        global_audio_speed: float = 1.0,
    ) -> AudioMixPlan:
        return build_audio_mix_plan(
            timeline,
            Path(output_dir) / self.output_audio_name,
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            bitrate=self.bitrate,
            clip_gain_db=self.clip_gain_db,
            background_audio=background_audio,
            background_gain_db=self.background_gain_db,
            output_duration_sec=output_duration_sec,
            pre_speed_duration_sec=pre_speed_duration_sec,
            global_audio_speed=global_audio_speed,
            clip_lowpass_hz=self.clip_lowpass_hz,
            clip_peak_normalize_dbfs=self.clip_peak_normalize_dbfs,
            clip_fade_in_ms=self.clip_fade_in_ms,
            clip_fade_out_ms=self.clip_fade_out_ms,
            clip_tail_pad_ms=self.clip_tail_pad_ms,
            clip_tail_cleanup=self.clip_tail_cleanup,
            clip_tail_cleanup_ms=self.clip_tail_cleanup_ms,
            clip_tail_cleanup_lowpass_hz=self.clip_tail_cleanup_lowpass_hz,
        )

    def render_plan(
        self,
        timeline: DubTimeline,
        source_video: str | Path,
        output_dir: str | Path,
        *,
        background_audio: str | Path | None = None,
        dub_subtitle: str | Path | None = None,
        burn_subtitles: bool = False,
    ) -> DubbingRenderPlan:
        audio_mix = self.audio_mix_plan(timeline, output_dir, background_audio=background_audio)
        compose = build_compose_video_plan(
            source_video,
            audio_mix.output_audio,
            Path(output_dir) / self.output_video_name,
            subtitle_path=dub_subtitle if burn_subtitles else None,
        )
        return DubbingRenderPlan(audio_mix=audio_mix, compose_video=compose, dub_subtitle=Path(dub_subtitle) if dub_subtitle else None)


def build_audio_mix_plan(
    timeline: DubTimeline,
    output_audio: str | Path,
    *,
    sample_rate_hz: int = 24000,
    channels: int = 2,
    bitrate: str = "192k",
    clip_gain_db: float = 0.0,
    background_audio: str | Path | None = None,
    background_gain_db: float = -18.0,
    output_duration_sec: float | None = None,
    pre_speed_duration_sec: float | None = None,
    global_audio_speed: float = 1.0,
    clip_lowpass_hz: int = 6800,
    clip_peak_normalize_dbfs: float | None = -3.0,
    clip_fade_in_ms: int = 5,
    clip_fade_out_ms: int = 220,
    clip_tail_pad_ms: int = 220,
    clip_tail_cleanup: bool = True,
    clip_tail_cleanup_ms: int = 420,
    clip_tail_cleanup_lowpass_hz: int = 3600,
) -> AudioMixPlan:
    clips: list[AudioClipPlacement] = []
    warnings: list[str] = list(timeline.warnings)
    for segment in timeline.segments:
        if segment.audio_path is None:
            warnings.append(f"{segment.segment_id}: skipped missing audio path")
            continue
        clips.append(
            AudioClipPlacement(
                segment_id=segment.segment_id,
                audio_path=segment.audio_path,
                start_sec=segment.dub_start_sec,
                end_sec=segment.dub_start_sec + segment.audio_duration_sec,
                gain_db=clip_gain_db,
            )
        )
    source_duration = max(timeline.duration_sec, max((clip.end_sec for clip in clips), default=0.0))
    duration = float(output_duration_sec) if output_duration_sec is not None else source_duration
    return AudioMixPlan(
        clips=tuple(clips),
        output_audio=Path(output_audio),
        duration_sec=duration,
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        bitrate=bitrate,
        pre_speed_duration_sec=pre_speed_duration_sec,
        global_audio_speed=global_audio_speed,
        background_audio=Path(background_audio) if background_audio else None,
        background_gain_db=background_gain_db,
        clip_lowpass_hz=clip_lowpass_hz,
        clip_peak_normalize_dbfs=clip_peak_normalize_dbfs,
        clip_fade_in_ms=clip_fade_in_ms,
        clip_fade_out_ms=clip_fade_out_ms,
        clip_tail_pad_ms=clip_tail_pad_ms,
        clip_tail_cleanup=clip_tail_cleanup,
        clip_tail_cleanup_ms=clip_tail_cleanup_ms,
        clip_tail_cleanup_lowpass_hz=clip_tail_cleanup_lowpass_hz,
        warnings=tuple(warnings),
    )


def build_dubbing_render_plan(
    timeline: DubTimeline,
    source_video: str | Path,
    output_dir: str | Path,
    *,
    background_audio: str | Path | None = None,
    dub_subtitle: str | Path | None = None,
    burn_subtitles: bool = False,
) -> DubbingRenderPlan:
    return DubbingRenderService().render_plan(
        timeline,
        source_video,
        output_dir,
        background_audio=background_audio,
        dub_subtitle=dub_subtitle,
        burn_subtitles=burn_subtitles,
    )
