from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from eistara.core.pipeline import StageContext, output_internal_path


def load_tts_segments(context: StageContext) -> list[dict[str, Any]]:
    """Load TTS segments from inline artifacts first, then persisted JSON."""

    segments = _coerce_segments(context.task.get("tts_segments") or context.artifacts.get("tts_segments"))
    if segments:
        return segments
    for value in (context.task.get("tts_segments_json"), context.artifacts.get("tts_segments_json")):
        segments = load_tts_segments_json(value)
        if segments:
            return segments
    return []


def load_tts_segments_json(value: Any) -> list[dict[str, Any]]:
    if not value:
        return []
    path = Path(value)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return []
    rows = data.get("segments") if isinstance(data, dict) else data
    return _coerce_segments(rows)


def write_tts_segments_json(output_dir: Path, segments: list[dict[str, Any]]) -> Path:
    path = output_internal_path(output_dir, "tts_segments.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _coerce_segments(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(segment) for segment in value if isinstance(segment, dict)]
