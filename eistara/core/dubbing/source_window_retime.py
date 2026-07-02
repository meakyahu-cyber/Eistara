from __future__ import annotations

from dataclasses import replace

from eistara.core.timeline import (
    DubTimeline,
    DubTimelineSegment,
    TimelineInput,
    TimelinePolicy,
    build_group_source_windows,
    segment_number_and_line,
)


SOURCE_WINDOW_OVERFLOW_EPSILON_SEC = 0.001
SOURCE_WINDOW_RETIME_TIER2_VIDEO_SPEED_MIN = 0.88
SOURCE_WINDOW_RETIME_TIER2_MAX_AUDIO_SPEED = 1.14


def source_window_stretch_info(
    inputs: list[TimelineInput],
    policy: TimelinePolicy,
    *,
    max_audio_speed: float,
    source_duration_sec: float | None = None,
) -> dict[str, object]:
    max_stretch = max(1.0, float(policy.source_window_stretch_max))
    manual_stretch = max(1.0, float(policy.source_window_stretch))
    initial_stretch = min(manual_stretch, max_stretch)
    info: dict[str, object] = {
        "source_window_stretch_enabled": bool(policy.uses_source_windows),
        "source_window_stretch_max": round(max_stretch, 3),
        "max_source_window_overflow_ratio": 1.0,
        "requested_source_window_stretch": round(manual_stretch, 3),
        "applied_source_window_stretch": round(initial_stretch, 3),
        "source_window_stretch_capped": manual_stretch > max_stretch + 0.001,
        "source_window_required_audio_speed": 1.0,
        "source_window_stretch_basis_segment_id": None,
        "source_window_stretch_overflow_sec": 0.0,
        "source_window_final_overflow_sec": 0.0,
        "source_window_retime_max_audio_speed": round(max(1.0, float(max_audio_speed)), 3),
    }
    if not policy.uses_source_windows:
        return info

    valid = [
        item
        for item in sorted(inputs, key=_timeline_input_sort_key)
        if item.source_end_sec >= item.source_start_sec
        and item.target_text.strip()
        and item.audio_duration_sec is not None
        and item.audio_duration_sec > 0
    ]
    if not valid:
        return info

    group_windows = build_group_source_windows(
        valid,
        max_gap_after_sec=policy.max_source_gap_sec,
        source_duration_sec=source_duration_sec,
    )
    group_items: dict[str, list[TimelineInput]] = {}
    for item in valid:
        group, _line_index = segment_number_and_line(item.segment_id)
        group_items.setdefault(group, []).append(item)

    ordered_groups = sorted(
        (group for group in group_items if group in group_windows),
        key=lambda group: (
            float(group_windows[group].source_start_sec),
            float(group_windows[group].source_end_sec),
            group,
        ),
    )
    max_ratio = 1.0
    basis_segment_id: str | None = None
    for group in ordered_groups:
        window = group_windows[group]
        source_start = float(window.source_start_sec)
        window_end = float(window.window_end_sec)
        window_duration = max(0.0, window_end - source_start)
        if window_duration <= 0:
            continue

        cursor: float | None = None
        group_audio_end = source_start
        last_segment_id = None
        for item in sorted(group_items[group], key=_timeline_input_sort_key):
            start = source_start if cursor is None else cursor + float(policy.line_gap_sec)
            end = start + float(item.audio_duration_sec or 0.0)
            cursor = end
            group_audio_end = max(group_audio_end, end)
            last_segment_id = item.segment_id
        ratio = max(1.0, (group_audio_end - source_start) / window_duration)
        if ratio > max_ratio:
            max_ratio = ratio
            basis_segment_id = str(last_segment_id) if last_segment_id is not None else group

    initial_policy = replace(policy, source_window_stretch=initial_stretch)
    initial_clip_info = source_window_clip_speed_info(
        inputs,
        initial_policy,
        max_audio_speed=max_audio_speed,
        source_duration_sec=source_duration_sec,
    )
    initial_overflow_sec = float(initial_clip_info.get("max_clip_overflow_after_speed_sec") or 0.0)
    requested = manual_stretch
    if initial_overflow_sec > 0.001:
        requested = max(
            requested,
            _find_source_window_stretch_for_clip_fit(
                inputs,
                policy,
                max_audio_speed=max_audio_speed,
                source_duration_sec=source_duration_sec,
                lower=initial_stretch,
                max_stretch=max_stretch,
            ),
        )
    applied = min(requested, max_stretch)
    applied_clip_info = (
        initial_clip_info
        if abs(applied - initial_stretch) <= 0.001
        else source_window_clip_speed_info(
            inputs,
            replace(policy, source_window_stretch=applied),
            max_audio_speed=max_audio_speed,
            source_duration_sec=source_duration_sec,
        )
    )
    required_audio_speed = max(1.0, float(applied_clip_info.get("max_required_clip_speed") or 1.0))
    final_overflow_sec = float(applied_clip_info.get("max_clip_overflow_after_speed_sec") or 0.0)
    info.update(
        {
            "max_source_window_overflow_ratio": round(max_ratio, 3),
            "requested_source_window_stretch": round(requested, 3),
            "applied_source_window_stretch": round(applied, 3),
            "source_window_stretch_capped": requested > applied + 0.001,
            "source_window_required_audio_speed": round(required_audio_speed, 3),
            "source_window_stretch_basis_segment_id": applied_clip_info.get("clip_speed_basis_segment_id")
            or initial_clip_info.get("clip_speed_basis_segment_id")
            or basis_segment_id,
            "source_window_stretch_overflow_sec": round(initial_overflow_sec, 3),
            "source_window_final_overflow_sec": round(final_overflow_sec, 3),
            "source_window_final_overflow_raw_sec": final_overflow_sec,
        }
    )
    return info


def source_window_needs_retime_tier2(stretch_info: dict[str, object]) -> bool:
    return float(
        stretch_info.get("source_window_final_overflow_raw_sec")
        if stretch_info.get("source_window_final_overflow_raw_sec") is not None
        else stretch_info.get("source_window_final_overflow_sec")
        or 0.0
    ) > SOURCE_WINDOW_OVERFLOW_EPSILON_SEC


def source_window_retime_tier_info(
    tier: int,
    tier1_info: dict[str, object] | None,
    selected_info: dict[str, object],
) -> dict[str, object]:
    info: dict[str, object] = {
        "retime_tier": int(tier),
        "retime_tier_reason": "tier1_local_window_overflow" if tier == 2 else "tier1_sufficient",
        "retime_tier2_video_speed_min": SOURCE_WINDOW_RETIME_TIER2_VIDEO_SPEED_MIN,
        "retime_tier2_max_audio_speed": SOURCE_WINDOW_RETIME_TIER2_MAX_AUDIO_SPEED,
    }
    if tier1_info is not None:
        info.update(
            {
                "retime_tier1_applied_source_window_stretch": tier1_info.get("applied_source_window_stretch"),
                "retime_tier1_source_window_stretch_max": tier1_info.get("source_window_stretch_max"),
                "retime_tier1_max_audio_speed": tier1_info.get("source_window_retime_max_audio_speed"),
                "retime_tier1_max_clip_overflow_after_speed_sec": tier1_info.get("source_window_final_overflow_sec"),
            }
        )
    info.update(
        {
            "selected_retime_tier_applied_source_window_stretch": selected_info.get("applied_source_window_stretch"),
            "selected_retime_tier_source_window_stretch_max": selected_info.get("source_window_stretch_max"),
            "selected_retime_tier_max_audio_speed": selected_info.get("source_window_retime_max_audio_speed"),
            "selected_retime_tier_max_clip_overflow_after_speed_sec": selected_info.get("source_window_final_overflow_sec"),
        }
    )
    return info


def source_window_clip_speed_info(
    inputs: list[TimelineInput],
    policy: TimelinePolicy,
    *,
    max_audio_speed: float,
    source_duration_sec: float | None = None,
) -> dict[str, object]:
    max_audio_speed = max(1.0, float(max_audio_speed))
    stretch = max(1.0, float(policy.source_window_stretch))
    valid = [
        item
        for item in sorted(inputs, key=_timeline_input_sort_key)
        if item.source_end_sec >= item.source_start_sec
        and item.target_text.strip()
        and item.audio_duration_sec is not None
        and item.audio_duration_sec > 0
    ]
    if not valid:
        return {
            "clip_speeds": {},
            "clip_required_speeds": {},
            "clip_starts": {},
            "clip_borrow_sec": {},
            "max_required_clip_speed": 1.0,
            "max_applied_clip_speed": 1.0,
            "clip_speeded_count": 0,
            "clip_audio_speed_capped_count": 0,
            "max_clip_overflow_after_speed_sec": 0.0,
            "source_window_borrow_enabled": bool(policy.source_window_borrow_enabled),
            "source_window_borrow_max_sec": float(policy.source_window_borrow_max_sec),
            "source_window_borrow_max_ratio": float(policy.source_window_borrow_max_ratio),
            "source_window_borrow_min_seam_sec": float(policy.source_window_borrow_min_seam_sec),
            "source_window_borrowed_count": 0,
            "source_window_borrowed_total_sec": 0.0,
            "source_window_borrowed_max_sec": 0.0,
            "clip_speed_basis_segment_id": None,
            "warnings": (),
            "max_audio_speed": round(max_audio_speed, 3),
        }

    group_windows = _source_window_group_windows(
        valid,
        policy.max_source_gap_sec,
        source_duration_sec=source_duration_sec,
    )
    group_items: dict[str, list[TimelineInput]] = {}
    for item in valid:
        group, _line_index = segment_number_and_line(item.segment_id)
        group_items.setdefault(group, []).append(item)
    ordered_groups = sorted(
        group_items,
        key=lambda group: (
            float(group_windows.get(group, {}).get("source_start_sec", 0.0)),
            float(group_windows.get(group, {}).get("source_end_sec", 0.0)),
            group,
        ),
    )

    clip_speeds: dict[str, float] = {}
    clip_required_speeds: dict[str, float] = {}
    clip_starts: dict[str, float] = {}
    clip_borrow_sec: dict[str, float] = {}
    warnings: list[str] = []
    max_required_speed = 1.0
    max_applied_speed = 1.0
    speeded_count = 0
    capped_count = 0
    max_overflow_after_speed = 0.0
    basis_segment_id: str | None = None
    previous_effective_end: float | None = None
    previous_group_was_natural_overflow = False
    borrowed_count = 0
    borrowed_total = 0.0
    borrowed_max = 0.0

    for group in ordered_groups:
        items = sorted(group_items[group], key=_timeline_input_sort_key)
        first_item = items[0]
        last_item = items[-1]
        window = group_windows.get(group, {})
        source_start = max(0.0, float(window.get("source_start_sec", first_item.source_start_sec)))
        source_end = max(source_start, float(window.get("source_end_sec", last_item.source_end_sec)))
        base_start = source_start * stretch
        window_end = float(window.get("window_end_sec", source_end)) * stretch
        raw_total_duration = sum(float(item.audio_duration_sec or 0.0) for item in items)
        if len(items) > 1:
            raw_total_duration += float(policy.line_gap_sec) * (len(items) - 1)
        natural_overflow_sec = max(0.0, base_start + raw_total_duration - window_end)

        borrow_sec = 0.0
        if (
            bool(policy.source_window_borrow_enabled)
            and natural_overflow_sec > 0.001
            and previous_effective_end is not None
            and not previous_group_was_natural_overflow
        ):
            seam_sec = max(0.0, float(policy.source_window_borrow_min_seam_sec))
            gap_after_seam = max(0.0, base_start - previous_effective_end - seam_sec)
            borrow_sec = min(
                natural_overflow_sec,
                max(0.0, float(policy.source_window_borrow_max_sec)),
                gap_after_seam * max(0.0, float(policy.source_window_borrow_max_ratio)),
            )
        dub_start = base_start - borrow_sec
        available_duration = max(0.001, window_end - dub_start)
        required_speed = max(1.0, raw_total_duration / available_duration)
        speed = min(max_audio_speed, required_speed)
        effective_total_duration = sum((float(item.audio_duration_sec or 0.0) / speed) for item in items)
        if len(items) > 1:
            effective_total_duration += float(policy.line_gap_sec) * (len(items) - 1)
        effective_group_end = dub_start + effective_total_duration
        overflow_after_speed = max(0.0, effective_group_end - window_end)
        if previous_effective_end is not None and dub_start < previous_effective_end - 0.001:
            warnings.append(
                f"{first_item.segment_id}: overlaps previous dub audio by {previous_effective_end - dub_start:.3f}s"
            )
        if overflow_after_speed > 0.001:
            warnings.append(
                f"{last_item.segment_id}: source window overflow by {overflow_after_speed:.3f}s"
            )

        cursor = dub_start
        for item in items:
            raw_duration = float(item.audio_duration_sec or 0.0)
            segment_id = str(item.segment_id)
            clip_speeds[segment_id] = speed
            clip_required_speeds[segment_id] = required_speed
            clip_starts[segment_id] = cursor
            clip_borrow_sec[segment_id] = borrow_sec if item == first_item else 0.0
            if speed > 1.001:
                speeded_count += 1
            if required_speed > max_audio_speed + 0.001:
                capped_count += 1
            cursor += (raw_duration / speed) + float(policy.line_gap_sec)

        if required_speed > max_required_speed:
            max_required_speed = required_speed
            basis_segment_id = str(last_item.segment_id)
        max_applied_speed = max(max_applied_speed, speed)
        max_overflow_after_speed = max(max_overflow_after_speed, overflow_after_speed)
        if borrow_sec > 0.001:
            borrowed_count += 1
            borrowed_total += borrow_sec
            borrowed_max = max(borrowed_max, borrow_sec)

        previous_effective_end = effective_group_end
        previous_group_was_natural_overflow = natural_overflow_sec > 0.001

    return {
        "clip_speeds": clip_speeds,
        "clip_required_speeds": clip_required_speeds,
        "clip_starts": clip_starts,
        "clip_borrow_sec": clip_borrow_sec,
        "max_required_clip_speed": round(max_required_speed, 3),
        "max_applied_clip_speed": round(max_applied_speed, 3),
        "clip_speeded_count": speeded_count,
        "clip_audio_speed_capped_count": capped_count,
        "max_clip_overflow_after_speed_sec": round(max_overflow_after_speed, 3),
        "source_window_borrow_enabled": bool(policy.source_window_borrow_enabled),
        "source_window_borrow_max_sec": float(policy.source_window_borrow_max_sec),
        "source_window_borrow_max_ratio": float(policy.source_window_borrow_max_ratio),
        "source_window_borrow_min_seam_sec": float(policy.source_window_borrow_min_seam_sec),
        "source_window_borrowed_count": borrowed_count,
        "source_window_borrowed_total_sec": round(borrowed_total, 3),
        "source_window_borrowed_max_sec": round(borrowed_max, 3),
        "clip_speed_basis_segment_id": basis_segment_id,
        "warnings": tuple(warnings),
        "max_audio_speed": round(max_audio_speed, 3),
    }


def source_window_timeline_from_clip_info(
    inputs: list[TimelineInput],
    policy: TimelinePolicy,
    clip_speed_info: dict[str, object],
) -> DubTimeline:
    speeds = dict(clip_speed_info.get("clip_speeds") or {})
    starts = dict(clip_speed_info.get("clip_starts") or {})
    segments: list[DubTimelineSegment] = []
    warnings = list(clip_speed_info.get("warnings") or ())
    for item in sorted(inputs, key=_timeline_input_sort_key):
        if item.source_end_sec < item.source_start_sec:
            warnings.append(f"{item.segment_id}: source end is before source start")
            continue
        if not item.target_text.strip():
            warnings.append(f"{item.segment_id}: skipped empty target text")
            continue
        if item.audio_duration_sec is None or item.audio_duration_sec <= 0:
            warnings.append(f"{item.segment_id}: skipped missing or empty audio duration")
            continue
        segment_id = str(item.segment_id)
        if segment_id not in starts:
            continue
        speed = max(1.0, float(speeds.get(segment_id, 1.0)))
        duration = float(item.audio_duration_sec or 0.0) / speed
        start = float(starts[segment_id])
        end = start + duration
        segments.append(
            DubTimelineSegment(
                segment_id=item.segment_id,
                source_start_sec=float(item.source_start_sec),
                source_end_sec=float(item.source_end_sec),
                dub_start_sec=round(start, 3),
                dub_end_sec=round(end, 3),
                target_text=item.target_text.strip(),
                source_text=item.source_text.strip(),
                speaker=item.speaker,
                audio_path=item.audio_path,
                audio_duration_sec=round(duration, 3),
            )
        )
    return DubTimeline(tuple(segments), tuple(warnings), tail_pad_sec=policy.tail_pad_sec, mode=policy.timeline_mode)


def _find_source_window_stretch_for_clip_fit(
    inputs: list[TimelineInput],
    policy: TimelinePolicy,
    *,
    max_audio_speed: float,
    source_duration_sec: float | None = None,
    lower: float,
    max_stretch: float,
) -> float:
    def overflow_at(stretch: float) -> float:
        info = source_window_clip_speed_info(
            inputs,
            replace(policy, source_window_stretch=max(1.0, float(stretch))),
            max_audio_speed=max_audio_speed,
            source_duration_sec=source_duration_sec,
        )
        return float(info.get("max_clip_overflow_after_speed_sec") or 0.0)

    low = max(1.0, float(lower))
    high = max(low, float(max_stretch))
    if overflow_at(high) > 0.001:
        high = max(high, low * 1.25)
        while high < 10.0 and overflow_at(high) > 0.001:
            high *= 1.25
        high = min(high, 10.0)

    if overflow_at(high) > 0.001:
        return high

    for _ in range(28):
        mid = (low + high) / 2
        if overflow_at(mid) > 0.001:
            low = mid
        else:
            high = mid
    return high


def _timeline_input_sort_key(item: TimelineInput) -> tuple[float, float, str, int, str]:
    number, line_index = segment_number_and_line(item.segment_id)
    return (float(item.source_start_sec), float(item.source_end_sec), number, line_index, str(item.segment_id))


def _source_window_group_windows(
    items: list[TimelineInput],
    max_gap_after_sec: float,
    *,
    source_duration_sec: float | None = None,
) -> dict[str, dict[str, float]]:
    windows = build_group_source_windows(
        items,
        max_gap_after_sec=max_gap_after_sec,
        source_duration_sec=source_duration_sec,
    )
    return {
        str(group): {
            "source_start_sec": window.source_start_sec,
            "source_end_sec": window.source_end_sec,
            "owned_gap_after_sec": window.owned_gap_after_sec,
            "window_end_sec": window.window_end_sec,
        }
        for group, window in windows.items()
    }
