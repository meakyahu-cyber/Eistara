from __future__ import annotations

from dataclasses import dataclass

from .text import normalize_subtitle_text, split_display_text, subtitle_visible_len
from .timecode import format_srt_timestamp


@dataclass(frozen=True, slots=True)
class SubtitleEvent:
    start_sec: float
    end_sec: float
    lines: tuple[str, ...]

    @property
    def timestamp(self) -> str:
        return format_srt_timestamp(self.start_sec, self.end_sec)


def build_display_events(
    start_sec: float,
    end_sec: float,
    columns: list[str],
    max_chars_by_column: dict[str, int],
) -> list[SubtitleEvent]:
    duration = max(0.001, float(end_sec) - float(start_sec))
    split_columns = [
        split_display_text(text, max_chars_by_column.get(str(index), max_chars_by_column.get("*", 20)))
        for index, text in enumerate(columns)
    ]
    event_count = max(len(chunks) for chunks in split_columns)
    weights = _event_timing_weights(split_columns, event_count)
    total_weight = sum(weights) or event_count
    cursor = float(start_sec)
    events: list[SubtitleEvent] = []
    for index in range(event_count):
        event_start = cursor
        event_end = float(end_sec) if index == event_count - 1 else cursor + duration * weights[index] / total_weight
        cursor = event_end
        lines = tuple(
            normalize_subtitle_text(_chunk_for_event(chunks, index))
            for chunks in split_columns
        )
        events.append(SubtitleEvent(event_start, event_end, tuple(line for line in lines if line)))
    return events


def render_srt(events: list[SubtitleEvent]) -> str:
    return "\n\n".join(
        f"{index}\n{event.timestamp}\n" + "\n".join(event.lines)
        for index, event in enumerate(events, 1)
    ).strip()


def _chunk_for_event(chunks: list[str], index: int) -> str:
    if not chunks:
        return ""
    if index < len(chunks):
        return chunks[index]
    return chunks[-1]


def _event_timing_weights(split_columns: list[list[str]], event_count: int) -> list[int]:
    if event_count <= 0:
        return []
    timing_chunks = split_columns[0] if split_columns else []
    weights = [
        max(1, subtitle_visible_len(_chunk_for_event(timing_chunks, index)))
        for index in range(event_count)
    ]
    return weights or [1] * event_count
