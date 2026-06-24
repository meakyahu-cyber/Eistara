from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass, field
from dataclasses import replace
from pathlib import Path

import pandas as pd

from eistara.core.delivery import ArtifactRole, SubtitleDeliveryGenerator
from eistara.core.jobs.models import StageName
from eistara.core.media import MediaInfo
from eistara.core.media import build_compose_video_plan
from eistara.core.media.validation import is_usable_media_file, remove_unusable_media_file
from eistara.core.pipeline import StageContext, StageResult, output_internal_path
from eistara.core.timeline import TimelineInput, TimelinePolicy, TimelinePreparationService, build_dub_timeline

from .audio import v1_processed_clip_duration_sec
from .models import AudioMixPlan
from .renderers import DubbingRenderer
from .service import DubbingRenderService


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
            tts_segments = context.task.get("tts_segments") or context.artifacts.get("tts_segments") or []
            if not tts_segments:
                return StageResult(status="skipped", skipped=True, warnings=["No dub_segments_json or tts_segments in task or artifacts"])
            output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
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
        output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
        generator = SubtitleDeliveryGenerator.from_config(context.config)
        inputs = generator.load_timeline_inputs_json(segments_path)
        inputs, duration_warnings = _apply_v1_processed_clip_durations(
            inputs,
            self.service,
            output_dir,
            job_dir=context.job_dir,
        )
        timeline = build_dub_timeline(inputs, self.timeline_policy)
        retime_info = _build_publish_retime_info(context, timeline, self.service, self.timeline_preparation.media_probe)
        audio_speed = float(retime_info["applied_audio_speed"])
        subtitle_timeline = _scale_timeline(timeline, audio_speed)
        tts_tasks = context.task.get("tts_tasks") or context.artifacts.get("tts_tasks")
        if tts_tasks:
            _write_v1_new_sub_times(Path(tts_tasks), subtitle_timeline)
        publish_retime_report = _write_publish_retime_report(output_dir, retime_info)
        dub_subtitles = generator.write_dub_timeline_subtitles(subtitle_timeline, output_dir)
        dub_subtitle = dub_subtitles[ArtifactRole.DUB_SUBTITLE]
        plan = self.service.audio_mix_plan(
            timeline,
            output_dir,
            background_audio=(
                context.task.get("background_audio") or context.artifacts.get("background_audio")
                if bool(context.task.get("audio_mix_include_background"))
                else None
            ),
            output_duration_sec=_retime_output_duration_sec(retime_info, subtitle_timeline.duration_sec),
            pre_speed_duration_sec=timeline.duration_sec,
            global_audio_speed=audio_speed,
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


_LINE_SEGMENT_RE = re.compile(r"^(?P<number>\d+(?:\.0)?)_(?P<line>\d+)$")


def _write_v1_new_sub_times(tts_tasks: Path, timeline) -> None:
    if not tts_tasks.exists():
        return
    df = pd.read_excel(tts_tasks)
    if "new_sub_times" not in df.columns:
        df["new_sub_times"] = None
    grouped: dict[str, list[tuple[int, list[float]]]] = {}
    for segment in timeline.segments:
        number, line_index = _segment_number_and_line(segment.segment_id)
        grouped.setdefault(number, []).append(
            (
                line_index,
                [round(float(segment.dub_start_sec), 3), round(float(segment.dub_end_sec), 3)],
            )
        )
    for index, row in df.iterrows():
        number = _number_text(row.get("number") or index + 1)
        values = [time_range for _line_index, time_range in sorted(grouped.get(number, []), key=lambda item: item[0])]
        if values:
            df.at[index, "new_sub_times"] = values
    df["timeline_end"] = round(float(timeline.duration_sec), 3) if timeline.segments else 0.0
    df.to_excel(tts_tasks, index=False)


def _segment_number_and_line(segment_id: str) -> tuple[str, int]:
    text = str(segment_id)
    match = _LINE_SEGMENT_RE.match(text)
    if match:
        return _number_text(match.group("number")), int(match.group("line"))
    return _number_text(text), 0


def _number_text(value) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


def _apply_v1_processed_clip_durations(
    inputs: list[TimelineInput],
    service: DubbingRenderService,
    output_dir: Path,
    *,
    job_dir: Path,
) -> tuple[list[TimelineInput], list[str]]:
    duration_plan = _v1_clip_processing_plan(service, output_dir)
    tail_pad_sec = max(0.0, float(service.clip_tail_pad_ms) / 1000)
    adjusted: list[TimelineInput] = []
    warnings: list[str] = []
    for item in inputs:
        audio_path = _resolve_audio_path(item.audio_path, output_dir, job_dir)
        if audio_path is not None and audio_path.exists():
            try:
                adjusted.append(
                    replace(
                        item,
                        audio_path=audio_path,
                        audio_duration_sec=v1_processed_clip_duration_sec(audio_path, duration_plan),
                    )
                )
                continue
            except Exception as exc:
                warnings.append(f"{item.segment_id}: failed to read processed clip duration: {exc}")
        if item.audio_duration_sec is None or item.audio_duration_sec <= 0 or tail_pad_sec <= 0:
            adjusted.append(item)
            continue
        adjusted.append(replace(item, audio_duration_sec=float(item.audio_duration_sec) + tail_pad_sec))
    return adjusted, warnings


def _v1_clip_processing_plan(service: DubbingRenderService, output_dir: Path) -> AudioMixPlan:
    return AudioMixPlan(
        clips=(),
        output_audio=output_dir / service.output_audio_name,
        duration_sec=0.0,
        sample_rate_hz=service.sample_rate_hz,
        channels=service.channels,
        bitrate=service.bitrate,
        clip_lowpass_hz=service.clip_lowpass_hz,
        clip_peak_normalize_dbfs=service.clip_peak_normalize_dbfs,
        clip_fade_in_ms=service.clip_fade_in_ms,
        clip_fade_out_ms=service.clip_fade_out_ms,
        clip_tail_pad_ms=service.clip_tail_pad_ms,
        clip_tail_cleanup=service.clip_tail_cleanup,
        clip_tail_cleanup_ms=service.clip_tail_cleanup_ms,
        clip_tail_cleanup_lowpass_hz=service.clip_tail_cleanup_lowpass_hz,
    )


def _resolve_audio_path(audio_path: Path | None, output_dir: Path, job_dir: Path) -> Path | None:
    if audio_path is None:
        return None
    path = Path(audio_path)
    if path.is_absolute() or path.exists():
        return path
    candidates = (
        job_dir / path,
        output_dir / path,
        output_dir / "audio" / "tmp" / path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def _build_publish_retime_info(context: StageContext, timeline, service: DubbingRenderService, media_probe) -> dict[str, object]:
    current_duration = float(timeline.duration_sec)
    source_duration = _source_duration_sec(context, media_probe)
    info: dict[str, object] = {
        "global_audio_speed_enabled": bool(service.publish_global_audio_speed),
        "source_duration_sec": round(source_duration, 3) if source_duration is not None else None,
        "original_dub_duration_sec": round(current_duration, 3),
        "target_video_speed_min": float(service.publish_target_video_speed_min),
        "max_audio_speed": float(service.publish_max_audio_speed),
        "short_video_speed_max": float(service.publish_short_video_speed_max),
        "short_video_speed_hard_max": float(service.publish_short_video_speed_hard_max),
        "wanted_video_speed": None,
        "video_speed_capped": False,
        "wanted_audio_speed": 1.0,
        "applied_audio_speed": 1.0,
        "audio_speed_capped": False,
        "projected_final_dub_duration_sec": round(current_duration, 3),
        "projected_video_speed": None,
        "reason": "",
    }

    if not service.publish_global_audio_speed:
        info["reason"] = "global_audio_speed_disabled"
        return _finalize_retime_info(info)
    target_video_speed_min = float(service.publish_target_video_speed_min)
    max_audio_speed = float(service.publish_max_audio_speed)
    if source_duration is None or source_duration <= 0 or current_duration <= 0 or target_video_speed_min <= 0 or max_audio_speed <= 1.0:
        info["reason"] = "invalid_duration_or_config"
        return _finalize_retime_info(info)

    target_dub_duration = source_duration / target_video_speed_min
    if current_duration <= target_dub_duration:
        info["projected_video_speed"] = round(source_duration / current_duration, 3)
        info["reason"] = "already_within_target"
        return _finalize_retime_info(info)

    wanted_speed = current_duration / target_dub_duration
    speed = min(max_audio_speed, wanted_speed)
    final_duration = current_duration / speed
    info.update(
        {
            "wanted_audio_speed": round(wanted_speed, 3),
            "applied_audio_speed": round(speed, 3),
            "audio_speed_capped": speed < wanted_speed,
            "projected_final_dub_duration_sec": round(final_duration, 3),
            "projected_video_speed": round(source_duration / final_duration, 3),
            "reason": "speeding_dub_to_reach_target_video_speed",
        }
    )
    return _finalize_retime_info(info)


def _finalize_retime_info(info: dict[str, object]) -> dict[str, object]:
    info["final_dub_duration_sec"] = info["projected_final_dub_duration_sec"]
    info["final_video_speed"] = info["projected_video_speed"]
    return info


def _timeline_audio_end_sec(timeline) -> float:
    return max((float(segment.dub_end_sec) for segment in timeline.segments), default=0.0)


def _retime_output_duration_sec(retime_info: dict[str, object], fallback: float) -> float:
    value = retime_info.get("final_dub_duration_sec") or retime_info.get("projected_final_dub_duration_sec")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return duration if duration > 0 else float(fallback)


def _source_duration_sec(context: StageContext, media_probe) -> float | None:
    if media_probe is None:
        return None
    for key in ("high_quality_audio", "raw_audio", "source_audio"):
        value = context.task.get(key) or context.artifacts.get(key)
        if not value:
            continue
        try:
            info: MediaInfo = media_probe.probe(str(value))
            duration = info.duration_sec or (info.audio.duration_sec if info.audio else None)
        except Exception:
            continue
        if duration is not None and duration > 0:
            return float(duration)
    return None


def _scale_timeline(timeline, speed: float):
    if speed <= 1.001:
        return timeline
    return replace(
        timeline,
        segments=tuple(
            replace(
                segment,
                dub_start_sec=round(float(segment.dub_start_sec) / speed, 3),
                dub_end_sec=round(float(segment.dub_end_sec) / speed, 3),
                audio_duration_sec=round(float(segment.audio_duration_sec) / speed, 3),
            )
            for segment in timeline.segments
        ),
        tail_pad_sec=round(float(timeline.tail_pad_sec) / speed, 3),
    )


def _write_publish_retime_report(output_dir: Path, info: dict[str, object]) -> Path:
    report_path = output_dir / "log" / "publish_retime.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


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
    source_bed_duck_volume: float = 0.12
    source_bed_lowpass_hz: int = 3600
    normalize_dub_audio: bool = True
    normalize_dub_audio_target_dbfs: float = -20.0
    stage: StageName = StageName.COMPOSE

    def run(self, context: StageContext) -> StageResult:
        source_video = context.task.get("source_video") or context.artifacts.get("source_video")
        if not source_video:
            return StageResult(status="skipped", skipped=True, warnings=["No source_video in task or artifacts"])
        output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
        output_dir.mkdir(parents=True, exist_ok=True)
        dub_audio = Path(context.task.get("dub_audio") or context.artifacts.get("dub_audio") or output_dir / self.service.output_audio_name)
        dub_subtitle = context.task.get("dub_subtitle") or context.artifacts.get("dub_subtitles") or output_dir / "output_dub.srt"
        burn_subtitles = self.burn_subtitles or bool(context.task.get("burn_subtitles"))
        video_speed = _video_speed_from_publish_retime(context) if self.video_retime else 1.0
        dub_audio, normalize_warnings, normalized_dub_audio = _prepare_v1_normalized_dub_audio(
            dub_audio,
            output_dir,
            enabled=self.normalize_dub_audio,
            target_dbfs=self.normalize_dub_audio_target_dbfs,
        )
        background_audio, background_warnings = _prepare_v1_background_bed(context, output_dir, self, video_speed)
        warnings = [
            *normalize_warnings,
            *background_warnings,
            *_video_speed_warnings(video_speed, self.video_speed_warn_min, self.video_speed_warn_max),
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


def _video_speed_from_publish_retime(context: StageContext) -> float:
    report = context.task.get("publish_retime_report") or context.artifacts.get("publish_retime_report")
    if not report:
        return 1.0
    try:
        data = json.loads(Path(report).read_text(encoding="utf-8-sig"))
    except Exception:
        return 1.0
    value = data.get("final_video_speed") or data.get("projected_video_speed")
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return speed if speed > 0 else 1.0


def _prepare_v1_normalized_dub_audio(
    dub_audio: Path,
    output_dir: Path,
    *,
    enabled: bool,
    target_dbfs: float,
) -> tuple[Path, list[str], Path | None]:
    if not enabled:
        return dub_audio, [], None
    if not dub_audio.exists():
        return dub_audio, [], None
    try:
        from pydub import AudioSegment
    except Exception as exc:
        return dub_audio, [f"dub audio normalization skipped: pydub is not available: {exc}"], None

    try:
        audio = AudioSegment.from_file(dub_audio)
    except Exception as exc:
        return dub_audio, [f"dub audio normalization skipped: failed to read {dub_audio}: {exc}"], None
    if len(audio) <= 0 or audio.dBFS == float("-inf"):
        return dub_audio, [f"dub audio normalization skipped: silent or empty audio: {dub_audio}"], None

    normalized = audio.apply_gain(float(target_dbfs) - audio.dBFS)
    normalized_path = output_dir / "normalized_dub.wav"
    normalized.export(normalized_path, format="wav")
    return normalized_path, [], normalized_path


def _video_speed_warnings(video_speed: float, warn_min: float, warn_max: float) -> list[str]:
    if abs(float(video_speed) - 1.0) <= 0.001:
        return []
    if float(video_speed) < float(warn_min) or float(video_speed) > float(warn_max):
        return [f"Video retime speed is {float(video_speed):.3f}x, outside preferred range {float(warn_min):.3f}-{float(warn_max):.3f}."]
    return []


def _prepare_v1_background_bed(
    context: StageContext,
    output_dir: Path,
    runner: ComposePlanStageRunner,
    video_speed: float,
) -> tuple[Path | None, list[str]]:
    background_file, source_mode, warnings = _resolve_v1_background_file(context, runner)
    if background_file is None:
        return None, warnings
    if not runner.background_ducking:
        return background_file, warnings

    tts_tasks = context.task.get("tts_tasks") or context.artifacts.get("tts_tasks")
    intervals = _v1_dub_intervals_ms(
        Path(tts_tasks) if tts_tasks else None,
        time_scale=video_speed,
        padding_ms=runner.background_duck_padding_ms,
        merge_gap_ms=runner.background_duck_merge_gap_ms,
    )
    if not intervals:
        return background_file, warnings

    try:
        from pydub import AudioSegment
    except Exception as exc:
        return background_file, [*warnings, f"background ducking skipped: pydub is not available: {exc}"]

    try:
        background = AudioSegment.from_file(background_file)
    except Exception as exc:
        return background_file, [*warnings, f"background ducking skipped: failed to read {background_file}: {exc}"]
    if len(background) <= 0:
        return background_file, warnings

    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_file = audio_dir / ("source_bed_ducked.wav" if source_mode else "background_ducked.wav")
    volume, lowpass_hz, filter_enabled = _v1_background_duck_settings(background, intervals, runner, source_mode)
    gain_db = 20 * math.log10(max(0.001, volume))
    transition_ms = max(0, int(runner.background_duck_transition_ms))

    output = AudioSegment.empty()
    cursor = 0
    for start_ms, end_ms in intervals:
        start_ms = min(max(0, int(start_ms)), len(background))
        end_ms = min(max(start_ms, int(end_ms)), len(background))
        if start_ms > cursor:
            output += background[cursor:start_ms]
        original_segment = background[start_ms:end_ms]
        segment = original_segment
        if filter_enabled and lowpass_hz > 0:
            segment = segment.low_pass_filter(lowpass_hz)
        segment = segment.apply_gain(gain_db)
        fade_ms = min(transition_ms, len(segment) // 2)
        if fade_ms > 0:
            start_mix = original_segment[:fade_ms].fade_out(fade_ms).overlay(segment[:fade_ms].fade_in(fade_ms))
            end_mix = segment[-fade_ms:].fade_out(fade_ms).overlay(original_segment[-fade_ms:].fade_in(fade_ms))
            segment = start_mix + segment[fade_ms : len(segment) - fade_ms] + end_mix
        output += segment
        cursor = end_ms
    if cursor < len(background):
        output += background[cursor:]

    output.export(output_file, format="wav")
    return output_file, warnings


def _resolve_v1_background_file(context: StageContext, runner: ComposePlanStageRunner) -> tuple[Path | None, bool, list[str]]:
    mode = str(runner.background_bed_mode).lower()
    if mode in {"source", "source_ducked", "raw", "raw_source"}:
        value = (
            context.task.get("high_quality_audio")
            or context.artifacts.get("high_quality_audio")
            or context.task.get("raw_audio")
            or context.artifacts.get("raw_audio")
        )
        warning = (
            "background_bed_mode uses the original source audio; "
            "this can leave source speech under the dub"
        )
        return (Path(value) if value else None), True, [warning] if value else []
    value = context.task.get("background_audio") or context.artifacts.get("background_audio")
    return (Path(value) if value else None), False, []


def _v1_background_duck_settings(background, intervals, runner: ComposePlanStageRunner, source_mode: bool) -> tuple[float, int, bool]:
    if source_mode:
        return float(runner.source_bed_duck_volume), int(runner.source_bed_lowpass_hz), True
    volume = float(runner.background_duck_volume)
    ducked_ms_total = sum(max(0, min(end, len(background)) - max(0, start)) for start, end in intervals)
    coverage = ducked_ms_total / len(background) if len(background) > 0 else 0.0
    if coverage >= float(runner.background_duck_high_coverage_threshold):
        high_volume = float(runner.background_duck_high_coverage_volume)
        if high_volume > volume:
            volume = high_volume
    return volume, int(runner.background_duck_lowpass_hz), bool(runner.background_duck_filter)


def _v1_dub_intervals_ms(tts_tasks: Path | None, *, time_scale: float, padding_ms: int, merge_gap_ms: int) -> list[list[int]]:
    if tts_tasks is None or not tts_tasks.exists():
        return []
    df = pd.read_excel(tts_tasks)
    intervals: list[list[int]] = []
    for _, row in df.iterrows():
        value = row.get("new_sub_times")
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        for start_time, end_time in _parse_time_ranges(value):
            mapped_start = float(start_time) * float(time_scale)
            mapped_end = float(end_time) * float(time_scale)
            start_ms = max(0, int(round(mapped_start * 1000)) - int(padding_ms))
            end_ms = max(start_ms, int(round(mapped_end * 1000)) + int(padding_ms))
            intervals.append([start_ms, end_ms])
    if not intervals:
        return []
    intervals.sort(key=lambda item: item[0])
    merged = [intervals[0]]
    for start_ms, end_ms in intervals[1:]:
        previous = merged[-1]
        if start_ms <= previous[1] + int(merge_gap_ms):
            previous[1] = max(previous[1], end_ms)
        else:
            merged.append([start_ms, end_ms])
    return merged


def _parse_time_ranges(value) -> list[list[float]]:
    if isinstance(value, str):
        value = re.sub(r"(?:np\.)?(?:float64|float32|float16|int64|int32|int16)\(([^()]+)\)", r"\1", value)
        value = ast.literal_eval(value)
    return [[float(start), float(end)] for start, end in value]
