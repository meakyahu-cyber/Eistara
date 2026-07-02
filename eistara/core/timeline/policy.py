from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimelinePolicy:
    lead_in_sec: float = 0.3
    line_gap_sec: float = 0.18
    row_gap_sec: float = 0.26
    tail_pad_sec: float = 0.5
    min_source_gap_sec: float = 0.12
    max_source_gap_sec: float = 6.0
    source_gap_scale: float = 0.45
    preserve_source_gaps: bool = True
    preserve_short_source_windows: bool = False
    timeline_mode: str = "cursor"
    source_window_stretch: float = 1.0
    source_window_stretch_max: float = 1.10
    source_window_borrow_enabled: bool = True
    source_window_borrow_max_sec: float = 0.60
    source_window_borrow_max_ratio: float = 0.50
    source_window_borrow_min_seam_sec: float = 0.12
    source_window_retime_tier2_enabled: bool = False

    @property
    def uses_source_windows(self) -> bool:
        return str(self.timeline_mode).strip().lower() in {"source_window", "source-windows", "source_windows"}

    def scaled_source_gap(self, previous_source_end_sec: float, next_source_start_sec: float) -> float:
        if not self.preserve_source_gaps:
            return 0.0
        raw_gap = max(0.0, float(next_source_start_sec) - float(previous_source_end_sec))
        if raw_gap < self.min_source_gap_sec:
            return 0.0
        scaled = raw_gap * max(0.0, self.source_gap_scale)
        return min(scaled, self.max_source_gap_sec)

    def inter_segment_gap(self, previous_source_end_sec: float, next_source_start_sec: float) -> float:
        scaled = self.scaled_source_gap(previous_source_end_sec, next_source_start_sec)
        return scaled if scaled > 0 else self.row_gap_sec
