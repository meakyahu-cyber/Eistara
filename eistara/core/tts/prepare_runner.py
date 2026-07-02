from __future__ import annotations

import re
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from eistara.core.jobs.models import StageName
from eistara.core.pipeline import StageContext, StageResult
from eistara.core.subtitle import parse_time_seconds
from eistara.core.tts.reference_audio import extract_reference_audio_segments
from eistara.core.tts.segments import load_tts_segments, write_tts_segments_json


@dataclass(slots=True)
class TtsPrepareStageRunner:
    audio_config: dict[str, Any] | None = None
    stage: StageName = StageName.TTS_PREPARE

    def run(self, context: StageContext) -> StageResult:
        segments = load_tts_segments(context)
        output_dir = _resolve_output_dir(context)
        if not segments:
            segments = _segments_from_v1_audio_subtitles(context, output_dir)
        if not segments:
            return StageResult(status="skipped", skipped=True, warnings=["No tts_segments or V1 audio subtitles in task or artifacts"])

        audio_dir = output_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        reference_audio_dir = audio_dir / "refers"
        reference_audio_dir.mkdir(parents=True, exist_ok=True)
        tts_tasks = audio_dir / "tts_tasks.xlsx"
        df = pd.DataFrame([_task_row(segment, index) for index, segment in enumerate(segments, 1)])
        df, micro_merge_report = _apply_micro_line_merges(df, self.audio_config or {}, output_dir)
        timing_error = _invalid_timing_error(context, output_dir, segments, df)
        if timing_error:
            return StageResult(
                status="failed",
                outputs={"tts_segments_count": len(segments)},
                warnings=[timing_error],
            )
        df.to_excel(tts_tasks, index=False)
        warnings = _extract_reference_audio(context, output_dir, reference_audio_dir, tts_tasks)

        normalized_segments = _normalized_segments_from_task_sheet(tts_tasks, output_dir)
        tts_segments_json = write_tts_segments_json(output_dir, normalized_segments)
        return StageResult(
            outputs={
                "tts_segments": normalized_segments,
                "tts_segments_json": str(tts_segments_json),
                "tts_segments_count": len(normalized_segments),
                "tts_tasks": str(tts_tasks),
                "reference_audio_dir": str(reference_audio_dir),
                **({"micro_tts_line_merge_report": str(micro_merge_report)} if micro_merge_report else {}),
            },
            warnings=warnings,
        )


def _task_row(segment: dict[str, Any], index: int) -> dict[str, Any]:
    segment_id = segment.get("id") or segment.get("number") or index
    start = _time_text(segment.get("start") or segment.get("start_sec") or segment.get("source_start_sec") or 0)
    end = _time_text(segment.get("end") or segment.get("end_sec") or segment.get("source_end_sec") or 0)
    duration = _duration(segment)
    text = _clean_tts_text(str(segment.get("text") or segment.get("target_text") or segment.get("target") or ""))
    origin = str(segment.get("source") or segment.get("origin") or "")
    lines = [_clean_tts_text(line) for line in _as_list(segment.get("lines"), default=[text]) if _clean_tts_text(line)]
    src_lines = [str(line) for line in _as_list(segment.get("src_lines"), default=[origin]) if str(line).strip()]
    return {
        "number": segment_id,
        "start_time": start,
        "end_time": end,
        "duration": duration,
        "text": text,
        "origin": origin,
        "lines": lines,
        "src_lines": src_lines,
    }


def _invalid_timing_error(
    context: StageContext,
    output_dir: Path,
    segments: list[dict[str, Any]],
    df: pd.DataFrame,
) -> str | None:
    if df.empty or not _expects_source_timing(context, output_dir, segments):
        return None
    if not _all_zero_time_windows(df):
        return None
    return "TTS timing invalid: all rows have zero source time windows despite timed subtitle input; refusing to continue."


def _expects_source_timing(context: StageContext, output_dir: Path, segments: list[dict[str, Any]]) -> bool:
    if _has_nonzero_segment_timing(segments):
        return True
    if _has_timed_rows(context.task.get("subtitle_rows")) or _has_timed_rows(context.artifacts.get("subtitle_rows")):
        return True
    for value in (
        context.task.get("subtitle_rows_json"),
        context.artifacts.get("subtitle_rows_json"),
    ):
        if _subtitle_rows_json_has_timing(value):
            return True
    for value in (
        context.task.get("audio_source_srt"),
        context.artifacts.get("audio_source_srt"),
        context.task.get("audio_translated_srt"),
        context.artifacts.get("audio_translated_srt"),
        context.task.get("source_srt"),
        context.artifacts.get("source_srt"),
        context.task.get("translated_srt"),
        context.artifacts.get("translated_srt"),
        output_dir / "audio" / "src_subs_for_audio.srt",
        output_dir / "audio" / "trans_subs_for_audio.srt",
    ):
        if _srt_has_timing(value):
            return True
    return False


def _all_zero_time_windows(df: pd.DataFrame) -> bool:
    if "start_time" not in df.columns or "end_time" not in df.columns:
        return False
    for _index, row in df.iterrows():
        if _row_has_positive_window(row.get("start_time"), row.get("end_time")):
            return False
    return True


def _has_nonzero_segment_timing(segments: list[dict[str, Any]]) -> bool:
    for segment in segments:
        start = segment.get("start", segment.get("start_sec", segment.get("source_start_sec")))
        end = segment.get("end", segment.get("end_sec", segment.get("source_end_sec")))
        if _row_has_positive_window(start, end):
            return True
        metadata = segment.get("metadata")
        if isinstance(metadata, dict) and _row_has_positive_window(metadata.get("start"), metadata.get("end")):
            return True
    return False


def _has_timed_rows(value: Any) -> bool:
    if not value:
        return False
    rows = value
    if isinstance(value, dict):
        rows = value.get("rows", [])
    if not isinstance(rows, list):
        return False
    for row in rows:
        if not isinstance(row, dict):
            continue
        if _row_has_positive_window(
            row.get("start_sec", row.get("start")),
            row.get("end_sec", row.get("end")),
        ):
            return True
    return False


def _subtitle_rows_json_has_timing(value: Any) -> bool:
    if not value:
        return False
    path = Path(value)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return _has_timed_rows(data)


def _srt_has_timing(value: Any) -> bool:
    if not value:
        return False
    path = Path(value)
    if not path.exists():
        return False
    try:
        rows = _parse_srt(path)
    except Exception:
        return False
    return any(_row_has_positive_window(row.get("start"), row.get("end")) for row in rows)


def _row_has_positive_window(start: Any, end: Any) -> bool:
    try:
        return parse_time_seconds(end) > parse_time_seconds(start)
    except Exception:
        return False


def _normalized_segment(segment: dict[str, Any], index: int, output_dir: Path) -> dict[str, Any]:
    segment_id = str(segment.get("id") or segment.get("number") or index)
    output_path = output_dir / "audio" / "tmp" / f"{segment_id}_0_temp.wav"
    result = dict(segment)
    result["id"] = segment_id
    result["output_path"] = str(output_path)
    result["text"] = _clean_tts_text(str(result.get("text") or segment.get("target_text") or segment.get("target") or ""))
    result["target_text"] = result["text"]
    return result


def _normalized_segments_from_task_sheet(tts_tasks: Path, output_dir: Path) -> list[dict[str, Any]]:
    df = pd.read_excel(tts_tasks)
    segments: list[dict[str, Any]] = []
    for row_index, row in df.iterrows():
        number = row.get("number") or row_index + 1
        lines = _as_list(row.get("lines"), default=[row.get("text", "")])
        src_lines = _as_list(row.get("src_lines"), default=[row.get("origin", "")])
        for line_index, line in enumerate(lines):
            text = _clean_tts_text(str(line))
            if not text:
                continue
            segment_id = f"{number}_{line_index}"
            segments.append(
                {
                    "id": segment_id,
                    "number": number,
                    "line_index": line_index,
                    "start": row.get("start_time", 0),
                    "end": row.get("end_time", 0),
                    "duration": row.get("duration", 0),
                    "text": text,
                    "target_text": text,
                    "source": str(src_lines[line_index]) if line_index < len(src_lines) else str(row.get("origin", "")),
                    "output_path": str(output_dir / "audio" / "tmp" / f"{number}_{line_index}_temp.wav"),
                }
            )
    return segments


def _apply_micro_line_merges(
    df: pd.DataFrame,
    audio_config: dict[str, Any],
    output_dir: Path,
) -> tuple[pd.DataFrame, Path | None]:
    if not _config_bool(audio_config, "merge_micro_lines", True):
        return df, None
    min_chars = _config_int(audio_config, "merge_micro_line_chars", 1)
    if min_chars < 1 or "lines" not in df.columns:
        return df, None

    result = df.copy()
    changes: list[dict[str, Any]] = []
    for index, row in result.iterrows():
        lines = [str(line).strip() for line in _as_list(row.get("lines")) if str(line).strip()]
        merged_lines, groups = _merge_micro_lines(lines, min_chars)
        if merged_lines == lines:
            continue
        result.at[index, "lines"] = merged_lines
        if "src_lines" in result.columns:
            result.at[index, "src_lines"] = _merge_source_lines_by_groups(_as_list(row.get("src_lines")), groups)
        changes.append(
            {
                "row_index": int(index),
                "number": _number_text(row.get("number") or index + 1),
                "before": lines,
                "after": merged_lines,
                "groups": groups,
            }
        )

    if not changes:
        return result, None

    report_path = output_dir / "log" / "micro_tts_line_merges.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "min_chars": min_chars,
                "changed_rows": len(changes),
                "changes": changes,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result, report_path


def _merge_micro_lines(lines: list[str], min_chars: int) -> tuple[list[str], list[list[int]]]:
    entries = [(str(line).strip(), [index]) for index, line in enumerate(lines) if str(line).strip()]
    if len(entries) <= 1 or min_chars < 1:
        return [text for text, _indexes in entries], [indexes for _text, indexes in entries]

    merged: list[tuple[str, list[int]]] = []
    pending_text = ""
    pending_indexes: list[int] = []

    for text, indexes in entries:
        if _count_tts_chars(text) <= min_chars:
            if merged:
                previous_text, previous_indexes = merged[-1]
                merged[-1] = (_join_tts_text(previous_text, text), previous_indexes + indexes)
            else:
                pending_text = _join_tts_text(pending_text, text)
                pending_indexes.extend(indexes)
            continue

        if pending_text:
            text = _join_tts_text(pending_text, text)
            indexes = pending_indexes + indexes
            pending_text = ""
            pending_indexes = []
        merged.append((text, indexes))

    if pending_text:
        if merged:
            previous_text, previous_indexes = merged[-1]
            merged[-1] = (_join_tts_text(previous_text, pending_text), previous_indexes + pending_indexes)
        else:
            merged.append((pending_text, pending_indexes))

    return [text for text, _indexes in merged], [indexes for _text, indexes in merged]


def _merge_source_lines_by_groups(source_lines: list[Any], groups: list[list[int]]) -> list[str]:
    normalized = [str(line).strip() for line in source_lines]
    merged: list[str] = []
    for group in groups:
        parts = [normalized[index] for index in group if index < len(normalized) and normalized[index]]
        merged.append(" ".join(parts))
    return merged


def _count_tts_chars(text: str) -> int:
    return len(re.sub(r"[\s\W_]+", "", str(text), flags=re.UNICODE))


def _join_tts_text(left: str, right: str) -> str:
    left = str(left).strip()
    right = str(right).strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left} {right}"


def _duration(segment: dict[str, Any]) -> float:
    raw = segment.get("duration") or segment.get("duration_sec") or segment.get("audio_duration_sec")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    try:
        return max(
            0.0,
            parse_time_seconds(segment.get("end") or segment.get("end_sec") or 0)
            - parse_time_seconds(segment.get("start") or segment.get("start_sec") or 0),
        )
    except Exception:
        return 0.0


def _time_text(value: object) -> str:
    seconds = parse_time_seconds(value)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{millis:03d}"


def _segments_from_v1_audio_subtitles(context: StageContext, output_dir: Path) -> list[dict[str, Any]]:
    trans_srt = _first_existing_path(
        context.task.get("audio_translated_srt"),
        context.artifacts.get("audio_translated_srt"),
        output_dir / "audio" / "trans_subs_for_audio.srt",
    )
    src_srt = _first_existing_path(
        context.task.get("audio_source_srt"),
        context.artifacts.get("audio_source_srt"),
        output_dir / "audio" / "src_subs_for_audio.srt",
    )
    if not trans_srt:
        return []
    translated = _parse_srt(trans_srt)
    sources = {item["number"]: item["text"] for item in _parse_srt(src_srt)} if src_srt else {}
    segments: list[dict[str, Any]] = []
    for item in translated:
        number = item["number"]
        start = item["start"]
        end = item["end"]
        segments.append(
            {
                "id": str(number),
                "number": number,
                "start": start,
                "end": end,
                "duration": max(0.0, parse_time_seconds(end) - parse_time_seconds(start)),
                "text": _clean_tts_text(item["text"]),
                "source": sources.get(number, ""),
            }
        )
    return segments


def _parse_srt(path: str | Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    text = Path(path).read_text(encoding="utf-8-sig")
    rows: list[dict[str, Any]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        try:
            number = int(lines[0])
        except ValueError:
            continue
        start, end = [part.strip() for part in lines[1].split("-->", 1)]
        rows.append({"number": number, "start": start, "end": end, "text": " ".join(lines[2:])})
    return rows


def _as_list(value: Any, *, default: list[Any] | None = None) -> list[Any]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return list(default or [])
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return list(default or [])
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return [text]
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        return [parsed]
    return [value]


def _number_text(value: Any) -> str:
    text = str(value)
    return text[:-2] if text.endswith(".0") else text


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


def _clean_tts_text(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", str(text)).strip()
    text = re.sub(r"锛圼^锛塢*锛?", "", text).strip()
    return text.replace("-", "")


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
    return None


def _extract_reference_audio(context: StageContext, output_dir: Path, reference_audio_dir: Path, tts_tasks: Path) -> list[str]:
    vocal_audio = _first_existing_path(
        context.task.get("vocal_audio"),
        context.artifacts.get("vocal_audio"),
        output_dir / "audio" / "vocal.mp3",
    )
    return extract_reference_audio_segments(
        output_dir,
        vocal_audio=vocal_audio,
        reference_audio_dir=reference_audio_dir,
        tts_tasks=tts_tasks,
    )


def _resolve_output_dir(context: StageContext) -> Path:
    output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
    if not output_dir.is_absolute():
        output_dir = context.job_dir / output_dir
    return output_dir
