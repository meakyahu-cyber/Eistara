from __future__ import annotations

"""Optional TTS pacing-quality retry pass.

This module is intentionally disabled by default. It is not part of the
source-window overflow strategy and it does not impose target durations.
Instead, it audits already generated TTS clips, estimates their active speech
rate against a per-job baseline, and retries clips that look unnaturally slow
or dragged, especially tiny clips such as one-character lines.

Keep this path conservative until it has more real-video coverage: retries add
provider calls, can be non-deterministic, and should only be kept when the new
audio is measurably closer to the baseline.
"""

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eistara.core.translation.pacing import count_chinese_chars, estimate_spoken_cost_units


@dataclass(frozen=True, slots=True)
class SegmentPacingStats:
    segment_id: str
    text: str
    audio_path: str
    spoken_units: int
    chinese_chars: int
    duration_sec: float = 0.0
    active_duration_sec: float = 0.0
    rate_units_per_sec: float = 0.0
    sec_per_unit: float = 0.0
    included_in_baseline: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "text": self.text,
            "audio_path": self.audio_path,
            "spoken_units": self.spoken_units,
            "chinese_chars": self.chinese_chars,
            "duration_sec": round(self.duration_sec, 3),
            "active_duration_sec": round(self.active_duration_sec, 3),
            "rate_units_per_sec": round(self.rate_units_per_sec, 3),
            "sec_per_unit": round(self.sec_per_unit, 3),
            "included_in_baseline": self.included_in_baseline,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class PacingRetryDecision:
    segment_id: str
    reason: str
    baseline_units_per_sec: float
    slow_threshold_units_per_sec: float
    expected_duration_sec: float
    drag_ratio: float
    original: SegmentPacingStats

    def to_dict(self) -> dict[str, Any]:
        return {
            "retry_kind": "tts_pacing_quality",
            "reason": self.reason,
            "baseline_units_per_sec": round(self.baseline_units_per_sec, 3),
            "slow_threshold_units_per_sec": round(self.slow_threshold_units_per_sec, 3),
            "expected_duration_sec": round(self.expected_duration_sec, 3),
            "drag_ratio": round(self.drag_ratio, 3),
            "original": self.original.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class PacingQualityPlan:
    enabled: bool
    baseline_units_per_sec: float | None = None
    baseline_sample_count: int = 0
    retry_decisions: dict[str, PacingRetryDecision] = field(default_factory=dict)
    stats_by_segment: dict[str, SegmentPacingStats] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "baseline_units_per_sec": (
                round(self.baseline_units_per_sec, 3) if self.baseline_units_per_sec is not None else None
            ),
            "baseline_sample_count": self.baseline_sample_count,
            "retry_count": len(self.retry_decisions),
            "warnings": list(self.warnings),
        }


def build_pacing_quality_plan(
    segments: list[dict[str, Any]],
    outputs: dict[str, str],
    audio_config: dict[str, Any],
) -> PacingQualityPlan:
    if not _config_bool(audio_config, "pacing_quality_check", False):
        return PacingQualityPlan(enabled=False)

    stats_by_segment: dict[str, SegmentPacingStats] = {}
    baseline_rates: list[float] = []
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        if not segment_id:
            continue
        audio_path = outputs.get(segment_id) or str(segment.get("audio_path") or "")
        stats = analyze_segment_pacing(segment_id, str(segment.get("text") or ""), audio_path, audio_config)
        include = _include_in_baseline(stats, segment)
        if include:
            stats = SegmentPacingStats(
                segment_id=stats.segment_id,
                text=stats.text,
                audio_path=stats.audio_path,
                spoken_units=stats.spoken_units,
                chinese_chars=stats.chinese_chars,
                duration_sec=stats.duration_sec,
                active_duration_sec=stats.active_duration_sec,
                rate_units_per_sec=stats.rate_units_per_sec,
                sec_per_unit=stats.sec_per_unit,
                included_in_baseline=True,
                error=stats.error,
            )
            baseline_rates.append(stats.rate_units_per_sec)
        stats_by_segment[segment_id] = stats

    min_samples = _config_int(audio_config, "pacing_baseline_min_samples", 3)
    if len(baseline_rates) < min_samples:
        return PacingQualityPlan(
            enabled=True,
            baseline_sample_count=len(baseline_rates),
            stats_by_segment=stats_by_segment,
            warnings=[f"pacing baseline skipped: only {len(baseline_rates)} usable samples"],
        )

    baseline = float(statistics.median(baseline_rates))
    slow_threshold = baseline * _config_float(audio_config, "pacing_slow_rate_ratio", 0.75)
    retry_decisions: dict[str, PacingRetryDecision] = {}
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        stats = stats_by_segment.get(segment_id)
        if stats is None:
            continue
        decision = _pacing_retry_decision(stats, baseline, slow_threshold, audio_config)
        if decision is not None:
            retry_decisions[segment_id] = decision

    return PacingQualityPlan(
        enabled=True,
        baseline_units_per_sec=baseline,
        baseline_sample_count=len(baseline_rates),
        retry_decisions=retry_decisions,
        stats_by_segment=stats_by_segment,
    )


def analyze_segment_pacing(
    segment_id: str,
    text: str,
    audio_path: str | Path,
    audio_config: dict[str, Any],
) -> SegmentPacingStats:
    text = str(text or "")
    path = Path(audio_path) if audio_path else Path()
    spoken_units = estimate_spoken_cost_units(text)
    chinese_chars = count_chinese_chars(text)
    if not audio_path:
        return SegmentPacingStats(segment_id, text, "", spoken_units, chinese_chars, error="missing audio path")
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except Exception as exc:
        return SegmentPacingStats(segment_id, text, str(path), spoken_units, chinese_chars, error=str(exc))
    try:
        audio = AudioSegment.from_file(path)
    except Exception as exc:
        return SegmentPacingStats(segment_id, text, str(path), spoken_units, chinese_chars, error=str(exc))
    duration_sec = len(audio) / 1000.0
    if len(audio) <= 0 or audio.dBFS == float("-inf"):
        return SegmentPacingStats(
            segment_id,
            text,
            str(path),
            spoken_units,
            chinese_chars,
            duration_sec=duration_sec,
            error="silent or empty audio",
        )
    min_silence_len = _config_int(audio_config, "pacing_min_silence_len_ms", 80)
    threshold_offset = _config_float(audio_config, "trim_silence_threshold_offset_db", 22.0)
    threshold_floor = _config_float(audio_config, "trim_silence_threshold_min_dbfs", -45.0)
    silence_thresh = max(audio.dBFS - threshold_offset, threshold_floor)
    nonsilent = detect_nonsilent(
        audio,
        min_silence_len=max(1, min_silence_len),
        silence_thresh=silence_thresh,
    )
    active_ms = sum(max(0, int(end - start)) for start, end in nonsilent)
    active_sec = active_ms / 1000.0
    rate = spoken_units / active_sec if spoken_units > 0 and active_sec > 0 else 0.0
    sec_per_unit = active_sec / spoken_units if spoken_units > 0 else 0.0
    return SegmentPacingStats(
        segment_id=segment_id,
        text=text,
        audio_path=str(path),
        spoken_units=spoken_units,
        chinese_chars=chinese_chars,
        duration_sec=duration_sec,
        active_duration_sec=active_sec,
        rate_units_per_sec=rate,
        sec_per_unit=sec_per_unit,
    )


def retry_improved_pacing(
    before: SegmentPacingStats,
    after: SegmentPacingStats,
    decision: PacingRetryDecision,
    audio_config: dict[str, Any],
) -> bool:
    if after.error or after.active_duration_sec <= 0 or after.rate_units_per_sec <= 0:
        return False
    baseline = max(0.001, decision.baseline_units_per_sec)
    before_error = abs(before.rate_units_per_sec - baseline) / baseline if before.rate_units_per_sec > 0 else 999.0
    after_error = abs(after.rate_units_per_sec - baseline) / baseline
    min_active_gain = _config_float(audio_config, "pacing_retry_min_active_gain_sec", 0.08)
    min_error_ratio = _config_float(audio_config, "pacing_retry_min_error_ratio", 0.92)
    if after.active_duration_sec <= before.active_duration_sec - min_active_gain:
        return True
    return after_error <= before_error * min_error_ratio


def _include_in_baseline(stats: SegmentPacingStats, segment: dict[str, Any]) -> bool:
    metadata = segment.get("metadata") if isinstance(segment.get("metadata"), dict) else {}
    request_duration_control = metadata.get("indextts_duration_control") or metadata.get("duration_control") or {}
    if isinstance(request_duration_control, dict) and _config_bool(request_duration_control, "enabled", False):
        return False
    return stats.error is None and stats.spoken_units >= 3 and stats.active_duration_sec >= 0.25 and 0.5 <= stats.rate_units_per_sec <= 12.0


def _pacing_retry_decision(
    stats: SegmentPacingStats,
    baseline: float,
    slow_threshold: float,
    audio_config: dict[str, Any],
) -> PacingRetryDecision | None:
    if stats.error or stats.spoken_units <= 0 or stats.active_duration_sec <= 0:
        return None
    expected_duration = stats.spoken_units / max(0.001, baseline)
    drag_ratio = stats.active_duration_sec / max(0.001, expected_duration)
    short_max_units = _config_int(audio_config, "pacing_short_max_units", 8)
    short_drag_ratio = _config_float(audio_config, "pacing_short_drag_ratio", 1.25)
    short_max_sec_per_unit = _config_float(audio_config, "pacing_short_max_sec_per_unit", 0.30)
    single_max_active_sec = _config_float(audio_config, "pacing_single_unit_max_active_sec", 0.70)
    slow_min_active_sec = _config_float(audio_config, "pacing_slow_min_active_sec", 1.0)

    reason: str | None = None
    if stats.spoken_units <= 2 and stats.active_duration_sec >= single_max_active_sec:
        reason = "single_or_tiny_segment_dragged"
    elif stats.spoken_units <= short_max_units and drag_ratio >= short_drag_ratio and stats.sec_per_unit >= short_max_sec_per_unit:
        reason = "short_segment_dragged"
    elif stats.active_duration_sec >= slow_min_active_sec and stats.rate_units_per_sec < slow_threshold:
        reason = "below_global_pacing_floor"
    if reason is None:
        return None
    return PacingRetryDecision(
        segment_id=stats.segment_id,
        reason=reason,
        baseline_units_per_sec=baseline,
        slow_threshold_units_per_sec=slow_threshold,
        expected_duration_sec=expected_duration,
        drag_ratio=drag_ratio,
        original=stats,
    )


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_int(config: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(config.get(key, default))
    except (TypeError, ValueError):
        return default


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default
