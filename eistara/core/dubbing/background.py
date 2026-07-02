from __future__ import annotations

import ast
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eistara.core.pipeline import StageContext, output_internal_path


def prepare_v1_background_bed(
    context: StageContext,
    output_dir: Path,
    runner: Any,
    video_speed: float,
    dub_audio: Path | None = None,
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
    settings = _v1_background_duck_settings(
        background,
        intervals,
        runner,
        source_mode,
        dub_audio=dub_audio,
        report_path=output_internal_path(output_dir, "background_duck_report.json"),
    )
    gain_db = settings.duck_gain_db
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
        if settings.filter_enabled and settings.lowpass_hz > 0:
            segment = segment.low_pass_filter(settings.lowpass_hz)
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

    _export_background_bed(output, output_file, makeup_gain_db=settings.makeup_gain_db)
    return output_file, warnings


@dataclass(frozen=True, slots=True)
class _BackgroundDuckSettings:
    duck_gain_db: float
    makeup_gain_db: float
    lowpass_hz: int
    filter_enabled: bool


def _resolve_v1_background_file(context: StageContext, runner: Any) -> tuple[Path | None, bool, list[str]]:
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


def _v1_background_duck_settings(
    background,
    intervals,
    runner: Any,
    source_mode: bool,
    *,
    dub_audio: Path | None = None,
    report_path: Path | None = None,
) -> _BackgroundDuckSettings:
    if source_mode:
        return _BackgroundDuckSettings(
            duck_gain_db=_volume_to_gain_db(float(runner.source_bed_duck_volume)),
            makeup_gain_db=0.0,
            lowpass_hz=int(runner.source_bed_lowpass_hz),
            filter_enabled=True,
        )
    coverage = _v1_duck_interval_coverage(background, intervals)
    if bool(runner.background_duck_adaptive):
        settings, report = _v1_adaptive_background_duck_settings(background, intervals, runner, dub_audio, coverage)
        if report_path is not None:
            _write_background_duck_report(report_path, report)
        return settings
    volume = float(runner.background_duck_volume)
    if coverage >= float(runner.background_duck_high_coverage_threshold):
        high_volume = float(runner.background_duck_high_coverage_volume)
        if high_volume > volume:
            volume = high_volume
    return _BackgroundDuckSettings(
        duck_gain_db=_volume_to_gain_db(volume),
        makeup_gain_db=0.0,
        lowpass_hz=int(runner.background_duck_lowpass_hz),
        filter_enabled=bool(runner.background_duck_filter),
    )


def _v1_adaptive_background_duck_settings(
    background,
    intervals: list[list[int]],
    runner: Any,
    dub_audio: Path | None,
    coverage: float,
) -> tuple[_BackgroundDuckSettings, dict[str, object]]:
    voice_under_db = float(runner.background_duck_target_under_voice_db)
    if coverage >= float(runner.background_duck_high_coverage_threshold):
        voice_under_db = float(runner.background_duck_high_coverage_under_voice_db)
    background_voice_dbfs = _audio_intervals_dbfs(background, intervals)
    background_full_dbfs = _finite_dbfs(background.dBFS)
    dub_voice_dbfs = _dub_voice_dbfs(dub_audio, intervals, fallback_dbfs=float(runner.normalize_dub_audio_target_dbfs))
    reference_background_dbfs = background_voice_dbfs if background_voice_dbfs is not None else background_full_dbfs
    if reference_background_dbfs is None or dub_voice_dbfs is None:
        fallback = _BackgroundDuckSettings(
            duck_gain_db=_volume_to_gain_db(float(runner.background_duck_volume)),
            makeup_gain_db=0.0,
            lowpass_hz=int(runner.background_duck_lowpass_hz),
            filter_enabled=bool(runner.background_duck_filter),
        )
        return fallback, {
            "mode": "fixed_fallback",
            "reason": "missing finite background or dub loudness",
            "coverage": coverage,
            "background_voice_dbfs": background_voice_dbfs,
            "background_full_dbfs": background_full_dbfs,
            "dub_voice_dbfs": dub_voice_dbfs,
            "duck_gain_db": fallback.duck_gain_db,
            "makeup_gain_db": fallback.makeup_gain_db,
            "lowpass_hz": fallback.lowpass_hz,
            "filter_enabled": fallback.filter_enabled,
        }

    target_background_voice_dbfs = dub_voice_dbfs - voice_under_db
    makeup_gain_db = max(0.0, target_background_voice_dbfs - reference_background_dbfs)
    makeup_gain_db = min(float(runner.background_duck_max_makeup_db), makeup_gain_db)
    after_makeup_dbfs = reference_background_dbfs + makeup_gain_db
    duck_gain_db = min(0.0, target_background_voice_dbfs - after_makeup_dbfs)
    filter_enabled = bool(runner.background_duck_filter)
    lowpass_hz = int(runner.background_duck_lowpass_hz)
    if bool(runner.background_duck_wideband_when_adaptive):
        filter_enabled = False
        lowpass_hz = 0
    settings = _BackgroundDuckSettings(
        duck_gain_db=duck_gain_db,
        makeup_gain_db=makeup_gain_db,
        lowpass_hz=lowpass_hz,
        filter_enabled=filter_enabled,
    )
    return settings, {
        "mode": "adaptive",
        "coverage": coverage,
        "target_under_voice_db": voice_under_db,
        "background_voice_dbfs": background_voice_dbfs,
        "background_full_dbfs": background_full_dbfs,
        "dub_voice_dbfs": dub_voice_dbfs,
        "target_background_voice_dbfs": target_background_voice_dbfs,
        "reference_background_dbfs": reference_background_dbfs,
        "after_makeup_background_dbfs": after_makeup_dbfs,
        "makeup_gain_db": makeup_gain_db,
        "duck_gain_db": duck_gain_db,
        "lowpass_hz": lowpass_hz,
        "filter_enabled": filter_enabled,
    }


def _v1_duck_interval_coverage(background, intervals: list[list[int]]) -> float:
    ducked_ms_total = sum(max(0, min(end, len(background)) - max(0, start)) for start, end in intervals)
    return ducked_ms_total / len(background) if len(background) > 0 else 0.0


def _audio_intervals_dbfs(audio, intervals: list[list[int]]) -> float | None:
    try:
        from pydub import AudioSegment
    except Exception:
        return None
    combined = AudioSegment.empty()
    for start_ms, end_ms in intervals:
        start_ms = min(max(0, int(start_ms)), len(audio))
        end_ms = min(max(start_ms, int(end_ms)), len(audio))
        if end_ms > start_ms:
            combined += audio[start_ms:end_ms]
    return _finite_dbfs(combined.dBFS) if len(combined) > 0 else None


def _dub_voice_dbfs(dub_audio: Path | None, intervals: list[list[int]], *, fallback_dbfs: float) -> float | None:
    if dub_audio is None or not Path(dub_audio).exists():
        return fallback_dbfs
    try:
        from pydub import AudioSegment
    except Exception:
        return fallback_dbfs
    try:
        audio = AudioSegment.from_file(dub_audio)
    except Exception:
        return fallback_dbfs
    measured = _audio_intervals_dbfs(audio, intervals)
    full = _finite_dbfs(audio.dBFS)
    return measured if measured is not None else full if full is not None else fallback_dbfs


def _finite_dbfs(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _volume_to_gain_db(volume: float) -> float:
    return 20 * math.log10(max(0.001, float(volume)))


def _write_background_duck_report(path: Path, report: dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _export_background_bed(audio, output_file: Path, *, makeup_gain_db: float) -> None:
    if abs(float(makeup_gain_db)) <= 0.001:
        audio.export(output_file, format="wav")
        return
    temp_file = output_file.with_name(f"{output_file.stem}.pre_makeup.wav")
    audio.export(temp_file, format="wav")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(temp_file),
        "-af",
        f"volume={float(makeup_gain_db):.6f}dB,alimiter=limit=0.891:level=false",
        "-c:a",
        "pcm_s16le",
        str(output_file),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception:
        fallback = audio.apply_gain(float(makeup_gain_db))
        if fallback.max_dBFS > -1.0:
            fallback = fallback.apply_gain(-1.0 - fallback.max_dBFS)
        fallback.export(output_file, format="wav")
    finally:
        try:
            temp_file.unlink()
        except OSError:
            pass


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
