from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from eistara.core.delivery import ArtifactRole, SubtitleDeliveryGenerator
from eistara.core.jobs.models import StageName
from eistara.core.media import build_compose_video_plan
from eistara.core.media.validation import is_usable_media_file, remove_unusable_media_file
from eistara.core.pipeline import StageContext, StageResult, output_internal_path, resolve_output_dir
from eistara.core.timeline import (
    TimelinePolicy,
    TimelinePreparationService,
)
from eistara.core.tts.segments import load_tts_segments

from .audio_mix_timing import build_audio_mix_timing_plan
from .background import prepare_v1_background_bed
from .compose import prepare_v1_normalized_dub_audio, video_speed_warnings
from .publish_retime import (
    video_speed_from_publish_retime,
    write_publish_retime_report,
)
from .renderers import DubbingRenderer
from .service import DubbingRenderService
from .v1_compat import apply_v1_processed_clip_durations, write_v1_new_sub_times


@dataclass(slots=True)
class AudioMixPlanStageRunner:
    service: DubbingRenderService = DubbingRenderService()
    renderer: DubbingRenderer | None = None
    timeline_preparation: TimelinePreparationService = field(default_factory=TimelinePreparationService)
    timeline_policy: TimelinePolicy = field(default_factory=TimelinePolicy)
    render_audio: bool = False
    stage: StageName = StageName.AUDIO_MIX

    def run(self, context: StageContext) -> StageResult:
        segments_path = (
            context.task.get("dub_segments_json")
            or context.task.get("timeline_json")
            or context.artifacts.get("dub_segments_json")
            or context.artifacts.get("timeline_json")
        )
        if not segments_path:
            tts_segments = load_tts_segments(context)
            if not tts_segments:
                return StageResult(status="skipped", skipped=True, warnings=["No dub_segments_json, tts_segments, or tts_segments_json in task or artifacts"])
            output_dir = resolve_output_dir(context)
            tts_outputs = (
                context.task.get("tts_outputs")
                or context.task.get("tts_segments_output")
                or context.artifacts.get("tts_outputs")
                or context.artifacts.get("tts_segments_output")
            )
            tts_durations = context.task.get("tts_durations") or context.artifacts.get("tts_durations")
            segments_path, _, prepare_warnings = self.timeline_preparation.write_segments(
                tts_segments,
                output_internal_path(output_dir, "dub_segments.json"),
                tts_outputs=tts_outputs,
                tts_durations=tts_durations,
            )
        else:
            prepare_warnings = []
        output_dir = resolve_output_dir(context)
        generator = SubtitleDeliveryGenerator.from_config(context.config)
        inputs = generator.load_timeline_inputs_json(segments_path)
        inputs, duration_warnings = apply_v1_processed_clip_durations(
            inputs,
            self.service,
            output_dir,
            job_dir=context.job_dir,
        )
        timing = build_audio_mix_timing_plan(
            inputs,
            self.timeline_policy,
            self.service,
            context,
            self.timeline_preparation.media_probe,
        )
        tts_tasks = context.task.get("tts_tasks") or context.artifacts.get("tts_tasks")
        if tts_tasks:
            write_v1_new_sub_times(Path(tts_tasks), timing.subtitle_timeline)
        publish_retime_report = write_publish_retime_report(output_dir, timing.retime_info)
        dub_subtitles = generator.write_dub_timeline_subtitles(timing.subtitle_timeline, output_dir)
        dub_subtitle = dub_subtitles[ArtifactRole.DUB_SUBTITLE]
        plan = self.service.audio_mix_plan(
            timing.placement_timeline,
            output_dir,
            background_audio=(
                context.task.get("background_audio") or context.artifacts.get("background_audio")
                if bool(context.task.get("audio_mix_include_background"))
                else None
            ),
            output_duration_sec=timing.output_duration_sec,
            pre_speed_duration_sec=timing.placement_timeline.duration_sec,
            global_audio_speed=float(timing.retime_info["applied_audio_speed"]),
            clip_speeds=timing.clip_speeds,
        )
        plan_path = output_internal_path(output_dir, "audio_mix_plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        render_result = None
        dub_audio_output: dict[str, str] = {}
        if self.renderer is not None and (self.render_audio or bool(context.task.get("render_audio"))):
            render_result = self.renderer.render_audio_mix(plan)
            if not render_result.ok:
                raise RuntimeError(render_result.stderr or render_result.stdout or "audio mix render failed")
            if not is_usable_media_file(plan.output_audio, require_audio=True):
                remove_unusable_media_file(plan.output_audio, require_audio=True)
                raise RuntimeError(f"audio mix render wrote unreadable audio: {plan.output_audio}")
            dub_audio_output["dub_audio"] = str(plan.output_audio)
        elif is_usable_media_file(plan.output_audio, require_audio=True):
            dub_audio_output["dub_audio"] = str(plan.output_audio)
        return StageResult(
            outputs={
                "audio_mix_plan": str(plan_path),
                "dub_segments_json": str(segments_path),
                "dub_subtitles": str(dub_subtitle),
                "dub_bilingual_subtitles": str(dub_subtitles.get(ArtifactRole.DUB_TARGET_SOURCE_SUBTITLE, "")),
                "publish_retime_report": str(publish_retime_report),
                "clip_count": len(plan.clips),
                **dub_audio_output,
                **(
                    {
                        "audio_render_command": list(render_result.command),
                        "audio_render_returncode": render_result.returncode,
                    }
                    if render_result is not None
                    else {}
                ),
            },
            warnings=[*prepare_warnings, *duration_warnings, *list(plan.warnings)],
        )

@dataclass(slots=True)
class ComposePlanStageRunner:
    service: DubbingRenderService = DubbingRenderService()
    renderer: DubbingRenderer | None = None
    render_video: bool = False
    burn_subtitles: bool = False
    video_encoder: str = ""
    audio_bitrate: str = "192k"
    audio_sample_rate_hz: int = 48000
    audio_channels: int = 2
    video_retime: bool = True
    video_retime_fps_mode: str = "cfr"
    video_speed_warn_min: float = 0.75
    video_speed_warn_max: float = 1.35
    source_fps: float = 30.0
    final_loudnorm: bool = False
    final_loudnorm_i: float = -16.0
    final_loudnorm_tp: float = -1.5
    final_loudnorm_lra: float = 4.5
    final_smooth: bool = False
    final_smooth_threshold_db: float = -30.0
    final_smooth_ratio: float = 4.0
    final_smooth_attack_ms: float = 6.0
    final_smooth_release_ms: float = 550.0
    final_smooth_makeup_db: float = 5.5
    background_ducking: bool = True
    background_bed_mode: str = "separated"
    background_duck_volume: float = 0.35
    background_duck_high_coverage_threshold: float = 0.85
    background_duck_high_coverage_volume: float = 0.35
    background_duck_padding_ms: int = 120
    background_duck_merge_gap_ms: int = 700
    background_duck_transition_ms: int = 600
    background_duck_filter: bool = True
    background_duck_lowpass_hz: int = 4200
    background_duck_adaptive: bool = True
    background_duck_target_under_voice_db: float = 16.0
    background_duck_high_coverage_under_voice_db: float = 14.0
    background_duck_max_makeup_db: float = 12.0
    background_duck_wideband_when_adaptive: bool = True
    source_bed_duck_volume: float = 0.12
    source_bed_lowpass_hz: int = 3600
    normalize_dub_audio: bool = True
    normalize_dub_audio_target_dbfs: float = -20.0
    stage: StageName = StageName.COMPOSE

    def run(self, context: StageContext) -> StageResult:
        source_video = context.task.get("source_video") or context.artifacts.get("source_video")
        if not source_video:
            return StageResult(status="skipped", skipped=True, warnings=["No source_video in task or artifacts"])
        output_dir = resolve_output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        dub_audio = Path(context.task.get("dub_audio") or context.artifacts.get("dub_audio") or output_dir / self.service.output_audio_name)
        dub_subtitle = context.task.get("dub_subtitle") or context.artifacts.get("dub_subtitles") or output_dir / "output_dub.srt"
        burn_subtitles = self.burn_subtitles or bool(context.task.get("burn_subtitles"))
        video_speed = video_speed_from_publish_retime(context) if self.video_retime else 1.0
        dub_audio, normalize_warnings, normalized_dub_audio = prepare_v1_normalized_dub_audio(
            dub_audio,
            output_dir,
            enabled=self.normalize_dub_audio,
            target_dbfs=self.normalize_dub_audio_target_dbfs,
        )
        background_audio, background_warnings = prepare_v1_background_bed(context, output_dir, self, video_speed, dub_audio)
        background_duck_report = output_internal_path(output_dir, "background_duck_report.json")
        warnings = [
            *normalize_warnings,
            *background_warnings,
            *video_speed_warnings(video_speed, self.video_speed_warn_min, self.video_speed_warn_max),
        ]
        compose = build_compose_video_plan(
            source_video,
            dub_audio,
            output_dir / self.service.output_video_name,
            subtitle_path=dub_subtitle if burn_subtitles else None,
            background_audio=background_audio,
            video_encoder=self.video_encoder,
            audio_bitrate=self.audio_bitrate,
            audio_sample_rate_hz=self.audio_sample_rate_hz,
            audio_channels=self.audio_channels,
            video_speed=video_speed,
            video_retime_fps_mode=self.video_retime_fps_mode,
            source_fps=self.source_fps,
            final_loudnorm=self.final_loudnorm,
            final_loudnorm_i=self.final_loudnorm_i,
            final_loudnorm_tp=self.final_loudnorm_tp,
            final_loudnorm_lra=self.final_loudnorm_lra,
            final_smooth=self.final_smooth,
            final_smooth_threshold_db=self.final_smooth_threshold_db,
            final_smooth_ratio=self.final_smooth_ratio,
            final_smooth_attack_ms=self.final_smooth_attack_ms,
            final_smooth_release_ms=self.final_smooth_release_ms,
            final_smooth_makeup_db=self.final_smooth_makeup_db,
            background_audio_speed=video_speed if abs(video_speed - 1.0) > 0.001 else 1.0,
            background_mix_duration="longest" if abs(video_speed - 1.0) > 0.001 else "first",
        )
        plan_path = output_internal_path(output_dir, "compose_plan.json")
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(
            json.dumps(
                {
                    "source_video": str(compose.source_video),
                    "dub_audio": str(compose.dub_audio),
                    "output_video": str(compose.output_video),
                    "subtitle_path": str(compose.subtitle_path) if compose.subtitle_path else None,
                    "external_subtitle_path": str(dub_subtitle),
                    "background_audio": str(compose.background_audio) if compose.background_audio else None,
                    "video_encoder": compose.video_encoder,
                    "audio_bitrate": compose.audio_bitrate,
                    "audio_sample_rate_hz": compose.audio_sample_rate_hz,
                    "audio_channels": compose.audio_channels,
                    "video_speed": compose.video_speed,
                    "video_retime_fps_mode": compose.video_retime_fps_mode,
                    "final_loudnorm": compose.final_loudnorm,
                    "final_smooth": compose.final_smooth,
                    "ffmpeg_args": list(compose.ffmpeg_args()),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        render_result = None
        dub_video_output: dict[str, str] = {}
        if self.renderer is not None and (self.render_video or bool(context.task.get("render_video"))):
            render_result = self.renderer.render_video(compose)
            if not render_result.ok:
                raise RuntimeError(render_result.stderr or render_result.stdout or "video render failed")
            if not is_usable_media_file(compose.output_video):
                remove_unusable_media_file(compose.output_video)
                raise RuntimeError(f"video render wrote unreadable media: {compose.output_video}")
            dub_video_output["dub_video"] = str(compose.output_video)
        elif is_usable_media_file(compose.output_video):
            dub_video_output["dub_video"] = str(compose.output_video)
        return StageResult(
            outputs={
                "compose_plan": str(plan_path),
                **({"normalized_dub_audio": str(normalized_dub_audio)} if normalized_dub_audio else {}),
                **({"background_duck_report": str(background_duck_report)} if background_duck_report.exists() else {}),
                **dub_video_output,
                **(
                    {
                        "video_render_command": list(render_result.command),
                        "video_render_returncode": render_result.returncode,
                    }
                    if render_result is not None
                    else {}
                ),
            },
            warnings=warnings,
        )
