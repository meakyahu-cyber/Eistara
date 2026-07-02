from __future__ import annotations

from typing import Any


SOURCE_AUDIO_DURATION_KEYS = ("high_quality_audio", "raw_audio", "source_audio")


def source_duration_sec(
    context: Any,
    media_probe: Any,
    *,
    keys: tuple[str, ...] = SOURCE_AUDIO_DURATION_KEYS,
) -> float | None:
    if media_probe is None:
        return None
    task = getattr(context, "task", {}) or {}
    artifacts = getattr(context, "artifacts", {}) or {}
    for key in keys:
        value = task.get(key) or artifacts.get(key)
        if not value:
            continue
        try:
            info = media_probe.probe(str(value))
            audio_info = getattr(info, "audio", None)
            duration = getattr(info, "duration_sec", None) or (
                getattr(audio_info, "duration_sec", None) if audio_info is not None else None
            )
        except Exception:
            continue
        if duration is not None and duration > 0:
            return float(duration)
    return None
