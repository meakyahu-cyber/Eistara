from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from eistara.core.timeline import TimelineInput, segment_group_id, segment_number_and_line

from .audio import v1_processed_clip_duration_sec
from .models import AudioMixPlan
from .service import DubbingRenderService


def write_v1_new_sub_times(tts_tasks: Path, timeline) -> None:
    if not tts_tasks.exists():
        return
    df = pd.read_excel(tts_tasks)
    if "new_sub_times" not in df.columns:
        df["new_sub_times"] = None
    grouped: dict[str, list[tuple[int, list[float]]]] = {}
    for segment in timeline.segments:
        number, line_index = segment_number_and_line(segment.segment_id)
        grouped.setdefault(number, []).append(
            (
                line_index,
                [round(float(segment.dub_start_sec), 3), round(float(segment.dub_end_sec), 3)],
            )
        )
    for index, row in df.iterrows():
        number = segment_group_id(row.get("number") or index + 1)
        values = [time_range for _line_index, time_range in sorted(grouped.get(number, []), key=lambda item: item[0])]
        if values:
            df.at[index, "new_sub_times"] = values
    df["timeline_end"] = round(float(timeline.duration_sec), 3) if timeline.segments else 0.0
    df.to_excel(tts_tasks, index=False)


def apply_v1_processed_clip_durations(
    inputs: list[TimelineInput],
    service: DubbingRenderService,
    output_dir: Path,
    *,
    job_dir: Path,
) -> tuple[list[TimelineInput], list[str]]:
    duration_plan = v1_clip_processing_plan(service, output_dir)
    tail_pad_sec = max(0.0, float(service.clip_tail_pad_ms) / 1000)
    tail_pad_counts_in_timeline = bool(service.clip_tail_pad_counts_in_timeline)
    adjusted: list[TimelineInput] = []
    warnings: list[str] = []
    for item in inputs:
        audio_path = resolve_audio_path(item.audio_path, output_dir, job_dir)
        if audio_path is not None and audio_path.exists():
            try:
                processed_duration = v1_processed_clip_duration_sec(audio_path, duration_plan)
                timeline_duration = (
                    processed_duration
                    if tail_pad_counts_in_timeline
                    else max(0.001, processed_duration - tail_pad_sec)
                )
                adjusted.append(
                    replace(
                        item,
                        audio_path=audio_path,
                        audio_duration_sec=timeline_duration,
                    )
                )
                continue
            except Exception as exc:
                warnings.append(f"{item.segment_id}: failed to read processed clip duration: {exc}")
        if item.audio_duration_sec is None or item.audio_duration_sec <= 0 or tail_pad_sec <= 0 or not tail_pad_counts_in_timeline:
            adjusted.append(item)
            continue
        adjusted.append(replace(item, audio_duration_sec=float(item.audio_duration_sec) + tail_pad_sec))
    return adjusted, warnings


def v1_clip_processing_plan(service: DubbingRenderService, output_dir: Path) -> AudioMixPlan:
    return AudioMixPlan(
        clips=(),
        output_audio=output_dir / service.output_audio_name,
        duration_sec=0.0,
        sample_rate_hz=service.sample_rate_hz,
        channels=service.channels,
        bitrate=service.bitrate,
        clip_lowpass_hz=service.clip_lowpass_hz,
        clip_peak_normalize_dbfs=service.clip_peak_normalize_dbfs,
        clip_fade_in_ms=service.clip_fade_in_ms,
        clip_fade_out_ms=service.clip_fade_out_ms,
        clip_tail_pad_ms=service.clip_tail_pad_ms,
        clip_tail_pad_counts_in_timeline=service.clip_tail_pad_counts_in_timeline,
        clip_tail_cleanup=service.clip_tail_cleanup,
        clip_tail_cleanup_ms=service.clip_tail_cleanup_ms,
        clip_tail_cleanup_lowpass_hz=service.clip_tail_cleanup_lowpass_hz,
    )


def resolve_audio_path(audio_path: Path | None, output_dir: Path, job_dir: Path) -> Path | None:
    if audio_path is None:
        return None
    path = Path(audio_path)
    if path.is_absolute() or path.exists():
        return path
    candidates = (
        job_dir / path,
        output_dir / path,
        output_dir / "audio" / "tmp" / path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path
