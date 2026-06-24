from __future__ import annotations

from collections.abc import Iterable
import re

from .models import DubTimeline, DubTimelineSegment, TimelineInput
from .policy import TimelinePolicy


class DubTimelineBuilder:
    def __init__(self, policy: TimelinePolicy | None = None):
        self.policy = policy or TimelinePolicy()

    def build(self, inputs: Iterable[TimelineInput]) -> DubTimeline:
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

            group = _segment_group(item.segment_id)
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

        return DubTimeline(tuple(segments), tuple(warnings), tail_pad_sec=self.policy.tail_pad_sec)


def build_dub_timeline(inputs: Iterable[TimelineInput], policy: TimelinePolicy | None = None) -> DubTimeline:
    return DubTimelineBuilder(policy).build(inputs)


_LINE_SEGMENT_RE = re.compile(r"^(?P<number>\d+(?:\.0)?)_(?P<line>\d+)$")


def _timeline_sort_key(item: TimelineInput) -> tuple[float, float, int, int, str]:
    number, line = _segment_number_and_line(item.segment_id)
    return (float(item.source_start_sec), float(item.source_end_sec), number, line, item.segment_id)


def _segment_group(segment_id: str) -> str:
    match = _LINE_SEGMENT_RE.match(str(segment_id))
    return match.group("number") if match else str(segment_id)


def _segment_number_and_line(segment_id: str) -> tuple[int, int]:
    match = _LINE_SEGMENT_RE.match(str(segment_id))
    if not match:
        return (2**31 - 1, 2**31 - 1)
    return (int(float(match.group("number"))), int(match.group("line")))
