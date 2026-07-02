from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable

from eistara.core.subtitle import parse_time_seconds


LINE_SEGMENT_RE = re.compile(r"^(?P<number>\d+(?:\.0)?)_(?P<line>\d+)$")


@dataclass(frozen=True, slots=True)
class SourceWindow:
    item_id: object
    source_start_sec: float
    source_end_sec: float
    source_duration_sec: float
    next_source_start_sec: float | None
    owned_gap_after_sec: float
    window_start_sec: float
    window_end_sec: float
    window_duration_sec: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "source_start_sec": round(self.source_start_sec, 3),
            "source_end_sec": round(self.source_end_sec, 3),
            "source_duration_sec": round(self.source_duration_sec, 3),
            "next_source_start_sec": round(self.next_source_start_sec, 3) if self.next_source_start_sec is not None else None,
            "owned_gap_after_sec": round(self.owned_gap_after_sec, 3),
            "window_start_sec": round(self.window_start_sec, 3),
            "window_end_sec": round(self.window_end_sec, 3),
            "window_duration_sec": round(self.window_duration_sec, 3),
        }


@dataclass(frozen=True, slots=True)
class _WindowSeed:
    item_id: object
    source_start_sec: float
    source_end_sec: float
    index: int


def segment_group_id(segment_id: object) -> str:
    text = _number_text(segment_id)
    match = LINE_SEGMENT_RE.match(text)
    return _number_text(match.group("number")) if match else text


def segment_number_and_line(segment_id: object) -> tuple[str, int]:
    text = _number_text(segment_id)
    match = LINE_SEGMENT_RE.match(text)
    if match:
        return _number_text(match.group("number")), int(match.group("line"))
    return text, 0


def segment_sort_key(segment_id: object) -> tuple[int, int, str]:
    text = _number_text(segment_id)
    match = LINE_SEGMENT_RE.match(text)
    if not match:
        return (2**31 - 1, 2**31 - 1, text)
    return (int(float(match.group("number"))), int(match.group("line")), text)


def build_source_windows(
    items: Iterable[Any],
    *,
    max_gap_after_sec: float = 6.0,
) -> dict[object, SourceWindow]:
    seeds = [_window_seed(item, index) for index, item in enumerate(items)]
    seeds = [seed for seed in seeds if seed is not None]
    seeds.sort(key=lambda seed: (seed.source_start_sec, seed.source_end_sec, seed.index))
    return _windows_from_seeds(seeds, max_gap_after_sec=max_gap_after_sec, source_duration_sec=None)


def build_group_source_windows(
    items: Iterable[Any],
    *,
    max_gap_after_sec: float = 6.0,
    source_duration_sec: float | None = None,
    group_key: str | None = None,
) -> dict[str, SourceWindow]:
    seeds: list[_WindowSeed] = []
    for index, item in enumerate(items):
        seed = _window_seed(item, index)
        if seed is None or seed.source_end_sec <= seed.source_start_sec:
            continue
        group_id = _group_id(item, seed.item_id, group_key)
        seeds.append(
            _WindowSeed(
                item_id=group_id,
                source_start_sec=seed.source_start_sec,
                source_end_sec=seed.source_end_sec,
                index=seed.index,
            )
        )

    grouped: dict[str, _WindowSeed] = {}
    for seed in seeds:
        group = str(seed.item_id)
        current = grouped.get(group)
        if current is None:
            grouped[group] = seed
            continue
        grouped[group] = _WindowSeed(
            item_id=group,
            source_start_sec=min(current.source_start_sec, seed.source_start_sec),
            source_end_sec=max(current.source_end_sec, seed.source_end_sec),
            index=min(current.index, seed.index),
        )

    ordered = sorted(
        grouped.values(),
        key=lambda seed: (seed.source_start_sec, seed.source_end_sec, str(seed.item_id)),
    )
    return _windows_from_seeds(ordered, max_gap_after_sec=max_gap_after_sec, source_duration_sec=source_duration_sec)


def _window_seed(item: Any, index: int) -> _WindowSeed | None:
    item_id = _value(item, "id", "number", "segment_id", default=index + 1)
    start = parse_time_seconds(_value(item, "start", "start_sec", "source_start_sec", default=0))
    end = parse_time_seconds(_value(item, "end", "end_sec", "source_end_sec", default=0))
    duration = _positive_float(_value(item, "duration_sec", "duration", default=None))
    if end <= start and duration is not None:
        end = start + duration
    if end < start:
        end = start
    return _WindowSeed(item_id=item_id, source_start_sec=start, source_end_sec=end, index=index)


def _windows_from_seeds(
    seeds: Iterable[_WindowSeed],
    *,
    max_gap_after_sec: float,
    source_duration_sec: float | None,
) -> dict[Any, SourceWindow]:
    ordered = list(seeds)
    result: dict[Any, SourceWindow] = {}
    max_gap = max(0.0, float(max_gap_after_sec))
    for index, seed in enumerate(ordered):
        next_start = ordered[index + 1].source_start_sec if index + 1 < len(ordered) else None
        if next_start is not None:
            raw_gap_after = max(0.0, float(next_start) - seed.source_end_sec)
        elif source_duration_sec is not None and source_duration_sec > seed.source_end_sec:
            raw_gap_after = max(0.0, float(source_duration_sec) - seed.source_end_sec)
        else:
            raw_gap_after = 0.0
        owned_gap_after = min(raw_gap_after, max_gap)
        window_start = seed.source_start_sec
        window_end = max(seed.source_end_sec, seed.source_end_sec + owned_gap_after)
        result[seed.item_id] = SourceWindow(
            item_id=seed.item_id,
            source_start_sec=seed.source_start_sec,
            source_end_sec=seed.source_end_sec,
            source_duration_sec=max(0.0, seed.source_end_sec - seed.source_start_sec),
            next_source_start_sec=next_start,
            owned_gap_after_sec=owned_gap_after,
            window_start_sec=window_start,
            window_end_sec=window_end,
            window_duration_sec=max(0.0, window_end - window_start),
        )
    return result


def _group_id(item: Any, item_id: object, group_key: str | None) -> str:
    if group_key:
        group = _value(item, group_key, default=None)
        if group is not None:
            return _number_text(group)
    return segment_group_id(item_id)


def _value(item: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        for key in keys:
            if key in item and item[key] is not None:
                return item[key]
        metadata = item.get("metadata")
        if isinstance(metadata, dict):
            for key in keys:
                if key in metadata and metadata[key] is not None:
                    return metadata[key]
        return default
    for key in keys:
        if hasattr(item, key):
            value = getattr(item, key)
            if value is not None:
                return value
    return default


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _number_text(value: object) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text
