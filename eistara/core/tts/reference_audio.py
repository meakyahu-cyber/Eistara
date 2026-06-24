from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def extract_reference_audio_segments(
    output_dir: str | Path,
    *,
    vocal_audio: str | Path | None = None,
    reference_audio_dir: str | Path | None = None,
    tts_tasks: str | Path | None = None,
) -> list[str]:
    output_path = Path(output_dir)
    source_audio = Path(vocal_audio) if vocal_audio else output_path / "audio" / "vocal.mp3"
    reference_dir = Path(reference_audio_dir) if reference_audio_dir else output_path / "audio" / "refers"
    task_sheet = Path(tts_tasks) if tts_tasks else output_path / "audio" / "tts_tasks.xlsx"

    if not source_audio.exists():
        return [f"reference audio extraction skipped: vocal_audio does not exist: {source_audio}"]
    if not task_sheet.exists():
        return [f"reference audio extraction skipped: tts_tasks does not exist: {task_sheet}"]

    reference_dir.mkdir(parents=True, exist_ok=True)
    existing_refers = [
        path
        for path in reference_dir.glob("*.wav")
        if path.name != "indextts_prompt.wav"
    ]
    if existing_refers and min(path.stat().st_mtime for path in existing_refers) >= source_audio.stat().st_mtime:
        return []

    try:
        import soundfile as sf
    except Exception as exc:
        return [f"reference audio extraction skipped: soundfile is not available: {exc}"]

    try:
        for path in reference_dir.glob("*.wav"):
            path.unlink()

        audio_data, sample_rate = sf.read(source_audio)
        tasks_df = pd.read_excel(task_sheet)
        for _, row in tasks_df.iterrows():
            number = _number_text(row["number"])
            start = _time_to_samples(row["start_time"], sample_rate)
            end = _time_to_samples(row["end_time"], sample_rate)
            sf.write(reference_dir / f"{number}.wav", audio_data[start:end], sample_rate)
    except Exception as exc:
        return [f"reference audio extraction failed: {exc}"]
    return []


def _time_to_samples(value: Any, sample_rate: int) -> int:
    text = str(value)
    hours, minutes, seconds = text.split(":")
    seconds, milliseconds = seconds.split(".") if "." in seconds else seconds.split(",") if "," in seconds else (seconds, "0")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds) + float(milliseconds) / 1000
    return int(total_seconds * sample_rate)


def _number_text(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text
