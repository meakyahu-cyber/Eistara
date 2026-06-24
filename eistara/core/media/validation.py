from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def is_usable_media_file(
    path: str | Path,
    *,
    require_audio: bool = False,
    min_duration_sec: float = 0.001,
    ffprobe_path: str = "ffprobe",
    timeout_sec: float = 20.0,
) -> bool:
    """Return True only when a media file exists and ffprobe can read it."""

    media_path = Path(path)
    try:
        if not media_path.is_file() or media_path.stat().st_size <= 0:
            return False
    except OSError:
        return False

    try:
        result = subprocess.run(
            (
                ffprobe_path,
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(media_path),
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_sec,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False

    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return False

    streams = list(data.get("streams") or [])
    if require_audio and not any(stream.get("codec_type") == "audio" for stream in streams):
        return False

    duration = _duration_sec(data)
    return duration is None or duration >= min_duration_sec


def remove_unusable_media_file(path: str | Path, **kwargs: Any) -> None:
    media_path = Path(path)
    if not media_path.exists() or is_usable_media_file(media_path, **kwargs):
        return
    try:
        media_path.unlink()
    except FileNotFoundError:
        pass


def _duration_sec(data: dict[str, Any]) -> float | None:
    format_data = dict(data.get("format") or {})
    duration = _optional_float(format_data.get("duration"))
    if duration is not None:
        return duration
    stream_durations = [
        value
        for value in (_optional_float(stream.get("duration")) for stream in list(data.get("streams") or []))
        if value is not None
    ]
    return max(stream_durations) if stream_durations else None


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
