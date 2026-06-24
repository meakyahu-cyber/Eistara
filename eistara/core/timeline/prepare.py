from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.media import MediaProbe
from eistara.core.subtitle import parse_time_seconds


@dataclass(slots=True)
class TimelinePreparationService:
    media_probe: MediaProbe | None = None

    def build_segments(
        self,
        tts_segments: list[dict[str, Any]],
        tts_outputs: dict[str, str] | None = None,
        tts_durations: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str]]:
        outputs = {str(key): value for key, value in (tts_outputs or {}).items()}
        durations_by_id = {str(key): value for key, value in (tts_durations or {}).items()}
        segments: list[dict[str, Any]] = []
        warnings: list[str] = []
        for index, item in enumerate(tts_segments, 1):
            segment_id = str(item.get("id") or item.get("segment_id") or index)
            audio_path = item.get("audio_path") or item.get("output_path") or outputs.get(segment_id)
            duration = item.get("audio_duration_sec") or item.get("duration") or item.get("audio_duration")
            if duration is None:
                duration = durations_by_id.get(segment_id)
            if duration is None and audio_path and self.media_probe is not None:
                try:
                    info = self.media_probe.probe(str(audio_path))
                    duration = info.duration_sec or (info.audio.duration_sec if info.audio else None)
                except Exception as exc:
                    warnings.append(f"{segment_id}: failed to probe audio duration: {exc}")
            if duration is None:
                warnings.append(f"{segment_id}: missing audio duration")
            source_start = item.get("source_start_sec", item.get("start_sec", item.get("start", 0)))
            source_end = item.get("source_end_sec", item.get("end_sec", item.get("end", 0)))
            source_text = item.get("source_text") or item.get("source") or item.get("Source") or (item.get("metadata") or {}).get("source") or ""
            target_text = item.get("target_text") or item.get("target") or item.get("text") or item.get("Translation") or ""
            segments.append(
                {
                    "id": segment_id,
                    "source_start_sec": parse_time_seconds(source_start),
                    "source_end_sec": parse_time_seconds(source_end),
                    "source": str(source_text),
                    "target": str(target_text),
                    "audio_path": str(audio_path) if audio_path else "",
                    "audio_duration_sec": float(duration) if duration is not None else None,
                }
            )
        return segments, warnings

    def write_segments(
        self,
        tts_segments: list[dict[str, Any]],
        output_path: str | Path,
        *,
        tts_outputs: dict[str, str] | None = None,
        tts_durations: dict[str, float] | None = None,
    ) -> tuple[Path, list[dict[str, Any]], list[str]]:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        segments, warnings = self.build_segments(tts_segments, tts_outputs=tts_outputs, tts_durations=tts_durations)
        path.write_text(
            json.dumps({"segments": segments, "warnings": warnings}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path, segments, warnings
