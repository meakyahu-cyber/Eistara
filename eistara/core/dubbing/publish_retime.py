from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from eistara.core.media import source_duration_sec
from eistara.core.pipeline import StageContext
from eistara.core.timeline import DubTimeline, TimelineInput

from .service import DubbingRenderService


def build_source_window_publish_retime_info(
    context: StageContext,
    subtitle_timeline,
    service: DubbingRenderService,
    media_probe,
    *,
    source_window_stretch_info: dict[str, object],
    clip_speed_info: dict[str, object],
) -> dict[str, object]:
    source_duration = source_duration_sec(context, media_probe)
    timeline_mode = str(getattr(subtitle_timeline, "mode", "source_window") or "source_window")
    stretch = max(1.0, float(source_window_stretch_info.get("applied_source_window_stretch") or 1.0))
    source_scaled_duration = source_duration * stretch if source_duration is not None and source_duration > 0 else None
    timeline_duration = timeline_audio_end_sec(subtitle_timeline) + max(
        0.0, float(getattr(subtitle_timeline, "tail_pad_sec", 0.0) or 0.0)
    )
    final_duration = max(timeline_duration, source_scaled_duration or 0.0)
    if final_duration <= 0:
        final_duration = timeline_duration
    final_video_speed = 1.0 / stretch if stretch > 0 else 1.0
    source_window_stretch_max = max(1.0, float(source_window_stretch_info.get("source_window_stretch_max") or 1.0))
    target_video_speed_min = min(
        float(service.publish_target_video_speed_min),
        1.0 / source_window_stretch_max if source_window_stretch_max > 0 else float(service.publish_target_video_speed_min),
    )
    max_required_speed = float(clip_speed_info.get("max_required_clip_speed") or 1.0)
    max_applied_speed = float(clip_speed_info.get("max_applied_clip_speed") or 1.0)
    effective_max_audio_speed = float(clip_speed_info.get("max_audio_speed") or service.publish_max_audio_speed)
    capped_count = int(clip_speed_info.get("clip_audio_speed_capped_count") or 0)
    borrowed_count = int(clip_speed_info.get("source_window_borrowed_count") or 0)
    total_duration_overflow = 0.0
    if source_scaled_duration is not None:
        total_duration_overflow = max(0.0, timeline_duration - source_scaled_duration)
    else:
        total_duration_overflow = float(clip_speed_info.get("max_clip_overflow_after_speed_sec") or 0.0)

    if not service.publish_global_audio_speed:
        reason = "clip_audio_speed_disabled"
    elif capped_count > 0:
        reason = "source_window_clip_speed_limited_by_cap"
    elif max_applied_speed > 1.001:
        reason = "per_clip_speed_for_source_window_fit"
    elif borrowed_count > 0:
        reason = "source_window_borrowed_previous_gap"
    elif stretch > 1.001:
        reason = "source_window_stretch_video_slowdown"
    else:
        reason = "source_window_preserves_source_duration"

    info: dict[str, object] = {
        "global_audio_speed_enabled": bool(service.publish_global_audio_speed),
        "clip_audio_speed_mode": "per_clip_source_window",
        "timeline_mode": timeline_mode,
        "source_duration_sec": round(source_duration, 3) if source_duration is not None else None,
        "original_dub_duration_sec": round(timeline_duration, 3),
        "target_video_speed_min": round(target_video_speed_min, 3),
        "max_audio_speed": round(effective_max_audio_speed, 3),
        "short_video_speed_max": float(service.publish_short_video_speed_max),
        "short_video_speed_hard_max": float(service.publish_short_video_speed_hard_max),
        "wanted_video_speed": round(final_video_speed, 3),
        "video_speed_capped": final_video_speed < target_video_speed_min - 0.001,
        "wanted_audio_speed": round(max_required_speed, 3),
        "applied_audio_speed": round(max_applied_speed, 3),
        "max_applied_clip_speed": round(max_applied_speed, 3),
        "max_required_clip_speed": round(max_required_speed, 3),
        "clip_speeded_count": int(clip_speed_info.get("clip_speeded_count") or 0),
        "clip_audio_speed_capped_count": capped_count,
        "clip_speed_basis_segment_id": clip_speed_info.get("clip_speed_basis_segment_id"),
        "source_window_borrow_enabled": bool(clip_speed_info.get("source_window_borrow_enabled")),
        "source_window_borrow_max_sec": float(clip_speed_info.get("source_window_borrow_max_sec") or 0.0),
        "source_window_borrow_max_ratio": float(clip_speed_info.get("source_window_borrow_max_ratio") or 0.0),
        "source_window_borrow_min_seam_sec": float(clip_speed_info.get("source_window_borrow_min_seam_sec") or 0.0),
        "source_window_borrowed_count": borrowed_count,
        "source_window_borrowed_total_sec": float(clip_speed_info.get("source_window_borrowed_total_sec") or 0.0),
        "source_window_borrowed_max_sec": float(clip_speed_info.get("source_window_borrowed_max_sec") or 0.0),
        "audio_speed_capped": capped_count > 0,
        "source_window_audio_speed_capped": capped_count > 0,
        "total_duration_overflow_sec": round(total_duration_overflow, 3),
        "max_clip_overflow_after_speed_sec": float(clip_speed_info.get("max_clip_overflow_after_speed_sec") or 0.0),
        "projected_final_dub_duration_sec": round(final_duration, 3),
        "projected_video_speed": round(final_video_speed, 3),
        "reason": reason,
    }
    return finalize_retime_info(info)


def build_publish_retime_info(
    context: StageContext,
    timeline,
    service: DubbingRenderService,
    media_probe,
    *,
    source_window_stretch: float = 1.0,
    source_window_required_audio_speed: float = 1.0,
) -> dict[str, object]:
    source_duration = source_duration_sec(context, media_probe)
    timeline_mode = str(getattr(timeline, "mode", "cursor") or "cursor")
    current_duration = retimed_timeline_duration_sec(
        timeline,
        1.0,
        source_duration=source_duration,
        source_window_stretch=source_window_stretch,
        timeline_mode=timeline_mode,
    )
    info: dict[str, object] = {
        "global_audio_speed_enabled": bool(service.publish_global_audio_speed),
        "timeline_mode": timeline_mode,
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
        "source_window_audio_speed_capped": False,
        "total_duration_overflow_sec": 0.0,
        "projected_final_dub_duration_sec": round(current_duration, 3),
        "projected_video_speed": None,
        "reason": "",
    }

    if not service.publish_global_audio_speed:
        info["reason"] = "global_audio_speed_disabled"
        return finalize_retime_info(info)
    target_video_speed_min = float(service.publish_target_video_speed_min)
    max_audio_speed = float(service.publish_max_audio_speed)
    if source_duration is None or source_duration <= 0 or current_duration <= 0 or target_video_speed_min <= 0 or max_audio_speed <= 1.0:
        info["reason"] = "invalid_duration_or_config"
        return finalize_retime_info(info)

    min_window_audio_speed = max(1.0, float(source_window_required_audio_speed))
    if (
        timeline_mode.lower() in {"source_window", "source-windows", "source_windows"}
        and current_duration <= source_duration
        and min_window_audio_speed <= 1.001
    ):
        info.update(
            {
                "projected_final_dub_duration_sec": round(source_duration, 3),
                "projected_video_speed": 1.0,
                "reason": "source_window_preserves_source_duration",
            }
        )
        return finalize_retime_info(info)

    target_dub_duration = source_duration / target_video_speed_min
    if current_duration <= target_dub_duration and min_window_audio_speed <= 1.001:
        info["projected_video_speed"] = round(source_duration / current_duration, 3)
        info["reason"] = "already_within_target"
        return finalize_retime_info(info)

    global_wanted_speed = find_audio_speed_for_duration(
        timeline,
        target_dub_duration,
        source_duration=source_duration,
        source_window_stretch=source_window_stretch,
        timeline_mode=timeline_mode,
        max_audio_speed=max_audio_speed,
    )
    wanted_speed = max(global_wanted_speed, min_window_audio_speed)
    speed = min(max_audio_speed, wanted_speed)
    final_duration = retimed_timeline_duration_sec(
        timeline,
        speed,
        source_duration=source_duration,
        source_window_stretch=source_window_stretch,
        timeline_mode=timeline_mode,
    )
    natural_video_speed = source_duration / final_duration
    projected_video_speed = max(target_video_speed_min, natural_video_speed)
    total_duration_overflow = max(0.0, final_duration - (source_duration / projected_video_speed))
    audio_speed_capped = speed < wanted_speed - 0.001 or (
        speed >= max_audio_speed - 0.001 and total_duration_overflow > 0.001
    )
    info.update(
        {
            "wanted_audio_speed": round(wanted_speed, 3),
            "applied_audio_speed": round(speed, 3),
            "audio_speed_capped": audio_speed_capped,
            "source_window_audio_speed_capped": min_window_audio_speed > speed + 0.001,
            "projected_final_dub_duration_sec": round(final_duration, 3),
            "projected_video_speed": round(projected_video_speed, 3),
            "video_speed_capped": projected_video_speed > natural_video_speed + 0.001,
            "total_duration_overflow_sec": round(total_duration_overflow, 3),
            "reason": (
                "speeding_dub_and_video_limited_by_caps"
                if total_duration_overflow > 0.001
                else "speeding_dub_for_source_window_fit"
                if min_window_audio_speed > global_wanted_speed + 0.001
                else "speeding_dub_to_reach_target_video_speed"
            ),
        }
    )
    return finalize_retime_info(info)


def finalize_retime_info(info: dict[str, object]) -> dict[str, object]:
    info["final_dub_duration_sec"] = info["projected_final_dub_duration_sec"]
    info["final_video_speed"] = info["projected_video_speed"]
    return info


def timeline_audio_end_sec(timeline) -> float:
    return max((float(segment.dub_end_sec) for segment in timeline.segments), default=0.0)


def retimed_timeline_duration_sec(
    timeline,
    speed: float,
    *,
    source_duration: float | None,
    source_window_stretch: float,
    timeline_mode: str,
) -> float:
    speed = max(1.0, float(speed))
    audio_end = max(
        (
            float(segment.dub_start_sec) + (float(segment.audio_duration_sec) / speed)
            for segment in timeline.segments
        ),
        default=0.0,
    )
    duration = audio_end + max(0.0, float(getattr(timeline, "tail_pad_sec", 0.0) or 0.0))
    if (
        source_duration is not None
        and source_duration > 0
        and timeline_mode.lower() in {"source_window", "source-windows", "source_windows"}
        and float(source_window_stretch) > 1.001
    ):
        duration = max(duration, source_duration * float(source_window_stretch))
    return float(duration)


def find_audio_speed_for_duration(
    timeline,
    target_duration: float,
    *,
    source_duration: float | None,
    source_window_stretch: float,
    timeline_mode: str,
    max_audio_speed: float,
) -> float:
    target_duration = float(target_duration)
    max_audio_speed = max(1.0, float(max_audio_speed))
    if target_duration <= 0:
        return max_audio_speed
    if (
        retimed_timeline_duration_sec(
            timeline,
            1.0,
            source_duration=source_duration,
            source_window_stretch=source_window_stretch,
            timeline_mode=timeline_mode,
        )
        <= target_duration
    ):
        return 1.0
    if (
        retimed_timeline_duration_sec(
            timeline,
            max_audio_speed,
            source_duration=source_duration,
            source_window_stretch=source_window_stretch,
            timeline_mode=timeline_mode,
        )
        > target_duration
    ):
        return max_audio_speed
    low = 1.0
    high = max_audio_speed
    for _ in range(32):
        mid = (low + high) / 2
        duration = retimed_timeline_duration_sec(
            timeline,
            mid,
            source_duration=source_duration,
            source_window_stretch=source_window_stretch,
            timeline_mode=timeline_mode,
        )
        if duration > target_duration:
            low = mid
        else:
            high = mid
    return high


def retime_output_duration_sec(retime_info: dict[str, object], fallback: float) -> float:
    value = retime_info.get("final_dub_duration_sec") or retime_info.get("projected_final_dub_duration_sec")
    try:
        duration = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return duration if duration > 0 else float(fallback)


def uniform_clip_speeds(inputs: list[TimelineInput], speed: float) -> dict[str, float]:
    speed = max(1.0, float(speed))
    return {
        str(item.segment_id): speed
        for item in inputs
        if item.audio_duration_sec is not None and item.audio_duration_sec > 0
    }


def speed_timeline_inputs_by_id(inputs: list[TimelineInput], speeds: dict[str, float]) -> list[TimelineInput]:
    if not speeds or all(float(speed) <= 1.001 for speed in speeds.values()):
        return inputs
    return [
        replace(
            item,
            audio_duration_sec=round(float(item.audio_duration_sec or 0.0) / max(1.0, float(speeds.get(str(item.segment_id), 1.0))), 3),
        )
        for item in inputs
    ]


def placement_timeline_for_raw_audio(subtitle_timeline: DubTimeline, raw_inputs: list[TimelineInput]) -> DubTimeline:
    raw_duration_by_id = {
        str(item.segment_id): float(item.audio_duration_sec or 0.0)
        for item in raw_inputs
        if item.audio_duration_sec is not None and item.audio_duration_sec > 0
    }
    return replace(
        subtitle_timeline,
        segments=tuple(
            replace(
                segment,
                dub_end_sec=round(float(segment.dub_start_sec) + raw_duration_by_id.get(str(segment.segment_id), segment.audio_duration_sec), 3),
                audio_duration_sec=raw_duration_by_id.get(str(segment.segment_id), segment.audio_duration_sec),
            )
            for segment in subtitle_timeline.segments
        ),
    )


def write_publish_retime_report(output_dir: Path, info: dict[str, object]) -> Path:
    report_path = output_dir / "log" / "publish_retime.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def video_speed_from_publish_retime(context: StageContext) -> float:
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
