from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from eistara.core.media import source_duration_sec
from eistara.core.pipeline import StageContext
from eistara.core.timeline import TimelineInput, TimelinePolicy, build_dub_timeline

from .publish_retime import (
    build_publish_retime_info,
    build_source_window_publish_retime_info,
    placement_timeline_for_raw_audio,
    retime_output_duration_sec,
    speed_timeline_inputs_by_id,
    uniform_clip_speeds,
)
from .service import DubbingRenderService
from .source_window_retime import (
    SOURCE_WINDOW_RETIME_TIER2_MAX_AUDIO_SPEED,
    SOURCE_WINDOW_RETIME_TIER2_VIDEO_SPEED_MIN,
    source_window_clip_speed_info,
    source_window_needs_retime_tier2,
    source_window_retime_tier_info,
    source_window_stretch_info as build_source_window_stretch_info,
    source_window_timeline_from_clip_info,
)


@dataclass(slots=True)
class AudioMixTimingPlan:
    subtitle_timeline: object
    placement_timeline: object
    retime_info: dict[str, object]
    clip_speeds: dict[str, float]
    output_duration_sec: float


def build_audio_mix_timing_plan(
    inputs: list[TimelineInput],
    timeline_policy: TimelinePolicy,
    service: DubbingRenderService,
    context: StageContext,
    media_probe,
) -> AudioMixTimingPlan:
    source_duration = source_duration_sec(context, media_probe)
    source_window_base_audio_speed = service.publish_max_audio_speed if service.publish_global_audio_speed else 1.0
    source_window_max_audio_speed = source_window_base_audio_speed
    source_window_policy = timeline_policy
    source_window_stretch_info = build_source_window_stretch_info(
        inputs,
        source_window_policy,
        max_audio_speed=source_window_max_audio_speed,
        source_duration_sec=source_duration,
    )
    if (
        timeline_policy.uses_source_windows
        and timeline_policy.source_window_retime_tier2_enabled
        and source_window_needs_retime_tier2(source_window_stretch_info)
    ):
        tier1_info = dict(source_window_stretch_info)
        source_window_policy = replace(
            timeline_policy,
            source_window_stretch_max=1.0 / SOURCE_WINDOW_RETIME_TIER2_VIDEO_SPEED_MIN,
        )
        source_window_max_audio_speed = (
            max(source_window_base_audio_speed, SOURCE_WINDOW_RETIME_TIER2_MAX_AUDIO_SPEED)
            if service.publish_global_audio_speed
            else 1.0
        )
        source_window_stretch_info = build_source_window_stretch_info(
            inputs,
            source_window_policy,
            max_audio_speed=source_window_max_audio_speed,
            source_duration_sec=source_duration,
        )
        source_window_stretch_info.update(source_window_retime_tier_info(2, tier1_info, source_window_stretch_info))
    else:
        source_window_stretch_info.update(source_window_retime_tier_info(1, None, source_window_stretch_info))

    applied_policy = (
        replace(source_window_policy, source_window_stretch=source_window_stretch_info["applied_source_window_stretch"])
        if source_window_policy.uses_source_windows
        else source_window_policy
    )
    timeline = build_dub_timeline(inputs, applied_policy)
    if applied_policy.uses_source_windows:
        clip_speed_info = source_window_clip_speed_info(
            inputs,
            applied_policy,
            max_audio_speed=source_window_max_audio_speed,
            source_duration_sec=source_duration,
        )
        clip_speeds = dict(clip_speed_info["clip_speeds"])
        subtitle_timeline = source_window_timeline_from_clip_info(inputs, applied_policy, clip_speed_info)
        retime_info = build_source_window_publish_retime_info(
            context,
            subtitle_timeline,
            service,
            media_probe,
            source_window_stretch_info=source_window_stretch_info,
            clip_speed_info=clip_speed_info,
        )
    else:
        retime_info = build_publish_retime_info(
            context,
            timeline,
            service,
            media_probe,
            source_window_stretch=float(source_window_stretch_info["applied_source_window_stretch"]),
            source_window_required_audio_speed=float(source_window_stretch_info["source_window_required_audio_speed"]),
        )
        audio_speed = float(retime_info["applied_audio_speed"])
        clip_speeds = uniform_clip_speeds(inputs, audio_speed)
        subtitle_timeline = build_dub_timeline(speed_timeline_inputs_by_id(inputs, clip_speeds), applied_policy)

    retime_info.update(source_window_stretch_info)
    placement_timeline = placement_timeline_for_raw_audio(subtitle_timeline, inputs)
    return AudioMixTimingPlan(
        subtitle_timeline=subtitle_timeline,
        placement_timeline=placement_timeline,
        retime_info=retime_info,
        clip_speeds=clip_speeds,
        output_duration_sec=retime_output_duration_sec(retime_info, subtitle_timeline.duration_sec),
    )
