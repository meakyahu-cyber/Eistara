from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.subtitle import build_display_events, render_srt
from eistara.core.timeline import DubTimeline, TimelineInput, TimelinePolicy, build_dub_timeline

from .profile import ArtifactRole, DeliveryProfile, SubtitleColumn, default_delivery_profile


@dataclass(frozen=True, slots=True)
class SubtitleRow:
    start_sec: float
    end_sec: float
    source: str
    target: str
    speaker: str = "SPEAKER_00"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "SubtitleRow":
        return cls(
            start_sec=float(data.get("start_sec", data.get("start", 0))),
            end_sec=float(data.get("end_sec", data.get("end", 0))),
            source=str(data.get("source") or data.get("Source") or ""),
            target=str(data.get("target") or data.get("Translation") or ""),
            speaker=_speaker_id(data.get("speaker", data.get("speaker_id"))),
        )


DEFAULT_DISPLAY_LIMITS = {
    SubtitleColumn.SOURCE: 42,
    SubtitleColumn.TARGET: 20,
}


class SubtitleDeliveryGenerator:
    def __init__(
        self,
        profile: DeliveryProfile | None = None,
        display_limits: dict[SubtitleColumn, int] | None = None,
    ):
        self.profile = profile or default_delivery_profile()
        self.display_limits = dict(DEFAULT_DISPLAY_LIMITS)
        if display_limits:
            self.display_limits.update(display_limits)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any] | None,
        profile: DeliveryProfile | None = None,
    ) -> "SubtitleDeliveryGenerator":
        subtitle_config = (config or {}).get("subtitle") if isinstance(config, dict) else {}
        subtitle_config = subtitle_config if isinstance(subtitle_config, dict) else {}
        display_limits: dict[SubtitleColumn, int] = {}
        source_limit = _positive_int(subtitle_config.get("display_source_max_chars_per_line"))
        target_limit = _positive_int(subtitle_config.get("display_max_chars_per_line"))
        if source_limit is not None:
            display_limits[SubtitleColumn.SOURCE] = source_limit
        if target_limit is not None:
            display_limits[SubtitleColumn.TARGET] = target_limit
        return cls(profile=profile, display_limits=display_limits)

    def write_source_timeline_subtitles(self, rows: list[SubtitleRow], output_dir: str | Path) -> dict[ArtifactRole, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        written: dict[ArtifactRole, Path] = {}
        for artifact in self.profile.artifacts:
            if artifact.kind != "subtitle" or artifact.timeline != "source" or not artifact.subtitle_columns:
                continue
            events = []
            for row in rows:
                columns = [self._column_text(row, column) for column in artifact.subtitle_columns]
                limits = {
                    str(index): self.display_limits[column]
                    for index, column in enumerate(artifact.subtitle_columns)
                }
                limits["*"] = self.display_limits[SubtitleColumn.TARGET]
                events.extend(build_display_events(row.start_sec, row.end_sec, columns, limits))
            path = artifact.path(output_path)
            path.write_text(render_srt(events), encoding="utf-8")
            written[artifact.role] = path
        return written

    def write_dub_timeline_subtitles(self, timeline: DubTimeline, output_dir: str | Path) -> dict[ArtifactRole, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        written: dict[ArtifactRole, Path] = {}
        for artifact in self.profile.artifacts:
            if artifact.kind != "subtitle" or artifact.timeline != "dub" or not artifact.subtitle_columns:
                continue
            events = []
            for segment in timeline.segments:
                if not segment.target_text.strip():
                    continue
                columns = [self._timeline_column_text(segment, column) for column in artifact.subtitle_columns]
                limits = {
                    str(index): self.display_limits[column]
                    for index, column in enumerate(artifact.subtitle_columns)
                }
                limits["*"] = self.display_limits[SubtitleColumn.TARGET]
                events.extend(build_display_events(segment.dub_start_sec, segment.dub_end_sec, columns, limits))
            path = artifact.path(output_path)
            path.write_text(render_srt(events), encoding="utf-8")
            written[artifact.role] = path
        return written

    def write_dub_timeline_subtitle(self, timeline: DubTimeline, output_dir: str | Path) -> Path:
        written = self.write_dub_timeline_subtitles(timeline, output_dir)
        return written[ArtifactRole.DUB_SUBTITLE]

    def write_dub_subtitle_from_json(
        self,
        path: str | Path,
        output_dir: str | Path,
        policy: TimelinePolicy | None = None,
    ) -> tuple[Path, DubTimeline]:
        inputs = self.load_timeline_inputs_json(path)
        timeline = build_dub_timeline(inputs, policy)
        return self.write_dub_timeline_subtitle(timeline, output_dir), timeline

    def load_rows_json(self, path: str | Path) -> list[SubtitleRow]:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("rows", [])
        if not isinstance(data, list):
            raise ValueError("Subtitle row input must be a list or an object with a rows list")
        return [SubtitleRow.from_mapping(dict(item)) for item in data]

    def load_timeline_inputs_json(self, path: str | Path) -> list[TimelineInput]:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("segments", data.get("rows", []))
        if not isinstance(data, list):
            raise ValueError("Timeline input must be a list or an object with a segments/rows list")
        return [_timeline_input_from_mapping(dict(item), index) for index, item in enumerate(data, 1)]

    def _column_text(self, row: SubtitleRow, column: SubtitleColumn) -> str:
        if column == SubtitleColumn.SOURCE:
            return row.source
        if column == SubtitleColumn.TARGET:
            return row.target
        raise ValueError(f"Unsupported subtitle column: {column}")

    def _timeline_column_text(self, segment, column: SubtitleColumn) -> str:
        if column == SubtitleColumn.SOURCE:
            return segment.source_text
        if column == SubtitleColumn.TARGET:
            return segment.target_text
        raise ValueError(f"Unsupported subtitle column: {column}")


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _timeline_input_from_mapping(data: dict[str, Any], index: int) -> TimelineInput:
    audio_path = data.get("audio_path") or data.get("audio") or data.get("AudioPath")
    duration = (
        data.get("audio_duration_sec")
        if data.get("audio_duration_sec") is not None
        else data.get("duration")
        if data.get("duration") is not None
        else data.get("audio_duration")
    )
    return TimelineInput(
        segment_id=str(data.get("segment_id") or data.get("id") or data.get("number") or index),
        source_start_sec=float(data.get("source_start_sec", data.get("start_sec", data.get("start", 0)))),
        source_end_sec=float(data.get("source_end_sec", data.get("end_sec", data.get("end", 0)))),
        target_text=str(data.get("target_text") or data.get("target") or data.get("Translation") or ""),
        source_text=str(data.get("source_text") or data.get("source") or data.get("Source") or (data.get("metadata") or {}).get("source") or ""),
        audio_path=Path(audio_path) if audio_path else None,
        audio_duration_sec=float(duration) if duration is not None else None,
        speaker=_speaker_id(data.get("speaker", data.get("speaker_id") or (data.get("metadata") or {}).get("speaker"))),
    )


def _speaker_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "SPEAKER_00"
