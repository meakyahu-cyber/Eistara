from __future__ import annotations

from collections.abc import Iterable

from .models import DubTimeline, DubTimelineSegment, TimelineInput
from .policy import TimelinePolicy
from .windows import build_group_source_windows, segment_group_id, segment_sort_key


class DubTimelineBuilder:
    def __init__(self, policy: TimelinePolicy | None = None):
        self.policy = policy or TimelinePolicy()

    def build(self, inputs: Iterable[TimelineInput]) -> DubTimeline:
        if self.policy.uses_source_windows:
            return self._build_source_window_timeline(inputs)

        ordered_inputs = sorted(inputs, key=_timeline_sort_key)
        segments: list[DubTimelineSegment] = []
        warnings: list[str] = []
        cursor = max(0.0, self.policy.lead_in_sec)
        previous_source_end: float | None = None
        previous_group: str | None = None

        for item in ordered_inputs:
            if item.source_end_sec < item.source_start_sec:
                warnings.append(f"{item.segment_id}: source end is before source start")
                continue
            if not item.target_text.strip():
                warnings.append(f"{item.segment_id}: skipped empty target text")
                continue
            if item.audio_duration_sec is None or item.audio_duration_sec <= 0:
                warnings.append(f"{item.segment_id}: skipped missing or empty audio duration")
                continue

            group = segment_group_id(item.segment_id)
            if previous_source_end is not None:
                if group == previous_group:
                    cursor += self.policy.line_gap_sec
                else:
                    cursor += self.policy.inter_segment_gap(previous_source_end, item.source_start_sec)

            dub_start = cursor
            dub_end = dub_start + float(item.audio_duration_sec)
            segments.append(
                DubTimelineSegment(
                    segment_id=item.segment_id,
                    source_start_sec=float(item.source_start_sec),
                    source_end_sec=float(item.source_end_sec),
                    dub_start_sec=dub_start,
                    dub_end_sec=dub_end,
                    target_text=item.target_text.strip(),
                    source_text=item.source_text.strip(),
                    speaker=item.speaker,
                    audio_path=item.audio_path,
                    audio_duration_sec=float(item.audio_duration_sec),
                )
            )
            cursor = dub_end
            previous_source_end = float(item.source_end_sec)
            previous_group = group

        return DubTimeline(tuple(segments), tuple(warnings), tail_pad_sec=self.policy.tail_pad_sec, mode=self.policy.timeline_mode)

    def _build_source_window_timeline(self, inputs: Iterable[TimelineInput]) -> DubTimeline:
        ordered_inputs = sorted(inputs, key=_timeline_sort_key)
        valid_inputs: list[TimelineInput] = []
        warnings: list[str] = []
        for item in ordered_inputs:
            if item.source_end_sec < item.source_start_sec:
                warnings.append(f"{item.segment_id}: source end is before source start")
                continue
            if not item.target_text.strip():
                warnings.append(f"{item.segment_id}: skipped empty target text")
                continue
            if item.audio_duration_sec is None or item.audio_duration_sec <= 0:
                warnings.append(f"{item.segment_id}: skipped missing or empty audio duration")
                continue
            valid_inputs.append(item)

        group_windows = _source_window_groups(valid_inputs, self.policy.max_source_gap_sec)
        segments: list[DubTimelineSegment] = []
        group_cursor: dict[str, float] = {}
        previous_group: str | None = None
        previous_max_end: float | None = None
        stretch = max(1.0, float(self.policy.source_window_stretch))

        for item in valid_inputs:
            group = segment_group_id(item.segment_id)
            source_start = max(0.0, float(item.source_start_sec))
            stretched_source_start = source_start * stretch
            if group == previous_group:
                dub_start = group_cursor[group] + self.policy.line_gap_sec
            else:
                dub_start = stretched_source_start

            dub_end = dub_start + float(item.audio_duration_sec)
            window = group_windows.get(group)
            window_end = window.window_end_sec if window is not None else None
            stretched_window_end = float(window_end) * stretch if window_end is not None else None
            if stretched_window_end is not None and dub_end > stretched_window_end + 0.001:
                warnings.append(
                    f"{item.segment_id}: source window overflow by {dub_end - stretched_window_end:.3f}s"
                )
            if previous_max_end is not None and dub_start < previous_max_end - 0.001:
                warnings.append(
                    f"{item.segment_id}: overlaps previous dub audio by {previous_max_end - dub_start:.3f}s"
                )
            segments.append(
                DubTimelineSegment(
                    segment_id=item.segment_id,
                    source_start_sec=source_start,
                    source_end_sec=float(item.source_end_sec),
                    dub_start_sec=round(dub_start, 3),
                    dub_end_sec=round(dub_end, 3),
                    target_text=item.target_text.strip(),
                    source_text=item.source_text.strip(),
                    speaker=item.speaker,
                    audio_path=item.audio_path,
                    audio_duration_sec=float(item.audio_duration_sec),
                )
            )
            group_cursor[group] = dub_end
            previous_group = group
            previous_max_end = max(previous_max_end or 0.0, dub_end)

        return DubTimeline(tuple(segments), tuple(warnings), tail_pad_sec=self.policy.tail_pad_sec, mode=self.policy.timeline_mode)


def build_dub_timeline(inputs: Iterable[TimelineInput], policy: TimelinePolicy | None = None) -> DubTimeline:
    return DubTimelineBuilder(policy).build(inputs)


def _timeline_sort_key(item: TimelineInput) -> tuple[float, float, int, int, str]:
    number, line, text = segment_sort_key(item.segment_id)
    return (float(item.source_start_sec), float(item.source_end_sec), number, line, text)


def _source_window_groups(items: Iterable[TimelineInput], max_gap_after_sec: float):
    return build_group_source_windows(items, max_gap_after_sec=max_gap_after_sec)
