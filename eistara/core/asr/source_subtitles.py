from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from eistara.core.delivery import SubtitleRow


DEFAULT_PAUSE_BOUNDARY_SEC = 0.65
DEFAULT_SOFT_MAX_WORDS = 28
DEFAULT_HARD_MAX_WORDS = 42
DEFAULT_MAX_DURATION_SEC = 18.0
DEFAULT_AUDIO_PAUSE_MIN_SEC = 0.45
DEFAULT_AUDIO_PAUSE_TOLERANCE_SEC = 0.25
DEFAULT_AUDIO_PAUSE_THRESHOLD_OFFSET_DB = 16.0

_TOKEN_RE = re.compile(r"[\w]+(?:[.'’][\w]+)*|[\u4e00-\u9fff]", re.UNICODE)
_SENTENCE_END_RE = re.compile(r"[.!?\u2026\u3002\uff01\uff1f][\"')\]\u300d\u300f]*\s*$")


@dataclass(frozen=True, slots=True)
class AudioPause:
    start_sec: float
    end_sec: float

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


@dataclass(frozen=True, slots=True)
class SourceSubtitleCleanupResult:
    rows: tuple[SubtitleRow, ...]
    report: dict[str, Any]


def detect_audio_pauses(
    audio_path: str | Path | None,
    *,
    provider_config: dict[str, Any] | None = None,
) -> tuple[tuple[AudioPause, ...] | None, dict[str, Any]]:
    config = provider_config or {}
    if not _config_bool(config.get("source_subtitle_audio_pause_enabled"), True):
        return None, {"enabled": False, "reason": "disabled"}
    if not audio_path:
        return None, {"enabled": True, "available": False, "reason": "missing_audio_path"}
    path = Path(audio_path)
    if not path.exists():
        return None, {"enabled": True, "available": False, "reason": "audio_path_not_found", "path": str(path)}

    min_pause_sec = _config_float(config.get("source_subtitle_audio_pause_min_sec"), DEFAULT_AUDIO_PAUSE_MIN_SEC)
    seek_step_ms = _config_int(config.get("source_subtitle_audio_pause_seek_step_ms"), 20)
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence

        audio = AudioSegment.from_file(path).set_channels(1)
        silence_threshold = _audio_silence_threshold(audio, config)
        ranges = detect_silence(
            audio,
            min_silence_len=max(1, int(min_pause_sec * 1000)),
            silence_thresh=silence_threshold,
            seek_step=max(1, int(seek_step_ms)),
        )
    except Exception as exc:
        return None, {
            "enabled": True,
            "available": False,
            "reason": "audio_pause_detection_failed",
            "path": str(path),
            "error": str(exc),
        }

    pauses = tuple(AudioPause(start / 1000, end / 1000) for start, end in ranges if end > start)
    return pauses, {
        "enabled": True,
        "available": True,
        "path": str(path),
        "pause_count": len(pauses),
        "min_pause_sec": min_pause_sec,
        "seek_step_ms": seek_step_ms,
        "silence_threshold_dbfs": round(float(silence_threshold), 3),
        "total_pause_sec": round(sum(pause.duration_sec for pause in pauses), 3),
    }


def normalize_source_subtitle_rows(
    rows: Iterable[SubtitleRow],
    *,
    provider_config: dict[str, Any] | None = None,
    audio_pauses: Iterable[AudioPause | dict[str, Any] | tuple[float, float]] | None = None,
) -> SourceSubtitleCleanupResult:
    config = provider_config or {}
    pause_boundary_sec = _config_float(
        config.get("source_subtitle_pause_boundary_sec"),
        DEFAULT_PAUSE_BOUNDARY_SEC,
    )
    soft_max_words = _config_int(
        config.get("source_subtitle_soft_max_words"),
        DEFAULT_SOFT_MAX_WORDS,
    )
    hard_max_words = _config_int(
        config.get("source_subtitle_hard_max_words"),
        DEFAULT_HARD_MAX_WORDS,
    )
    max_duration_sec = _config_float(
        config.get("source_subtitle_max_duration_sec"),
        DEFAULT_MAX_DURATION_SEC,
    )
    audio_pause_tolerance_sec = _config_float(
        config.get("source_subtitle_audio_pause_tolerance_sec"),
        DEFAULT_AUDIO_PAUSE_TOLERANCE_SEC,
    )
    audio_pause_list = _coerce_audio_pauses(audio_pauses) if audio_pauses is not None else None

    ordered = _clean_rows(rows)
    input_word_counts = [_word_count(row.source) for row in ordered]
    merged: list[SubtitleRow] = []
    current: SubtitleRow | None = None
    current_words = 0
    boundary_reasons: dict[str, int] = {}

    for index, row in enumerate(ordered):
        if current is None:
            current = row
            current_words = _word_count(row.source)
        else:
            current = SubtitleRow(
                start_sec=current.start_sec,
                end_sec=max(current.end_sec, row.end_sec),
                source=_append_source_text(current.source, row.source),
                target="",
            )
            current_words = _word_count(current.source)

        next_row = ordered[index + 1] if index + 1 < len(ordered) else None
        reason = _boundary_reason(
            current,
            current_words,
            next_row,
            pause_boundary_sec=pause_boundary_sec,
            soft_max_words=soft_max_words,
            hard_max_words=hard_max_words,
            max_duration_sec=max_duration_sec,
            audio_pauses=audio_pause_list,
            audio_pause_tolerance_sec=audio_pause_tolerance_sec,
        )
        if reason:
            merged.append(current)
            boundary_reasons[reason] = boundary_reasons.get(reason, 0) + 1
            current = None
            current_words = 0

    if current is not None:
        merged.append(current)
        boundary_reasons["end_of_input"] = boundary_reasons.get("end_of_input", 0) + 1

    output_word_counts = [_word_count(row.source) for row in merged]
    report = {
        "input_rows": len(ordered),
        "output_rows": len(merged),
        "input_avg_words": round(sum(input_word_counts) / len(input_word_counts), 3) if input_word_counts else 0.0,
        "input_median_words": round(float(median(input_word_counts)), 3) if input_word_counts else 0.0,
        "output_avg_words": round(sum(output_word_counts) / len(output_word_counts), 3) if output_word_counts else 0.0,
        "output_median_words": round(float(median(output_word_counts)), 3) if output_word_counts else 0.0,
        "word_level_input": _looks_word_level(input_word_counts),
        "pause_boundary_sec": pause_boundary_sec,
        "soft_max_words": soft_max_words,
        "hard_max_words": hard_max_words,
        "max_duration_sec": max_duration_sec,
        "audio_pause_tolerance_sec": audio_pause_tolerance_sec,
        "audio_pause_count": len(audio_pause_list) if audio_pause_list is not None else None,
        "audio_pause_boundaries_enabled": audio_pause_list is not None,
        "boundary_reasons": boundary_reasons,
    }
    return SourceSubtitleCleanupResult(tuple(merged), report)


def _clean_rows(rows: Iterable[SubtitleRow]) -> list[SubtitleRow]:
    cleaned: list[SubtitleRow] = []
    previous_end = 0.0
    for row in sorted(rows, key=lambda item: (item.start_sec, item.end_sec, item.source)):
        text = _normalize_space(row.source)
        if not text:
            continue
        start = max(0.0, float(row.start_sec))
        end = max(start, float(row.end_sec))
        if end == start:
            continue
        if start < previous_end and end <= previous_end:
            continue
        cleaned.append(SubtitleRow(start_sec=start, end_sec=end, source=text, target=""))
        previous_end = max(previous_end, end)
    return cleaned


def _boundary_reason(
    current: SubtitleRow,
    current_words: int,
    next_row: SubtitleRow | None,
    *,
    pause_boundary_sec: float,
    soft_max_words: int,
    hard_max_words: int,
    max_duration_sec: float,
    audio_pauses: tuple[AudioPause, ...] | None,
    audio_pause_tolerance_sec: float,
) -> str:
    if next_row is None:
        return "end_of_input"

    next_gap = max(0.0, float(next_row.start_sec) - float(current.end_sec))
    duration = max(0.0, float(current.end_sec) - float(current.start_sec))
    if (
        next_gap >= pause_boundary_sec
        and _audio_pause_between(current.end_sec, next_row.start_sec, audio_pauses, audio_pause_tolerance_sec)
    ):
        return "audio_pause"
    if audio_pauses is None and next_gap >= pause_boundary_sec:
        return "source_pause"
    if _is_sentence_end(current.source) and current_words >= 4:
        return "sentence_end"
    if current_words >= hard_max_words:
        return "hard_word_limit"
    if current_words >= soft_max_words and (next_gap >= 0.18 or duration >= max_duration_sec):
        return "soft_word_limit"
    if duration >= max_duration_sec and current_words >= max(8, soft_max_words // 2):
        return "duration_limit"
    return ""


def _append_source_text(left: str, right: str) -> str:
    left = _normalize_space(left)
    right = _normalize_space(right)
    if not left:
        return right
    if not right:
        return left
    if right.lower().startswith(left.lower()):
        return right

    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    max_overlap = min(len(left_tokens), len(right_tokens), 12)
    for size in range(max_overlap, 0, -1):
        if [token.lower() for token in left_tokens[-size:]] == [token.lower() for token in right_tokens[:size]]:
            suffix = " ".join(right_tokens[size:]).strip()
            if not suffix:
                return left
            return _normalize_space(f"{left} {suffix}")
    return _normalize_space(f"{left} {right}")


def _audio_pause_between(
    previous_end_sec: float,
    next_start_sec: float,
    audio_pauses: tuple[AudioPause, ...] | None,
    tolerance_sec: float,
) -> bool:
    if audio_pauses is None:
        return False
    start = min(float(previous_end_sec), float(next_start_sec)) - max(0.0, float(tolerance_sec))
    end = max(float(previous_end_sec), float(next_start_sec)) + max(0.0, float(tolerance_sec))
    return any(pause.end_sec >= start and pause.start_sec <= end for pause in audio_pauses)


def _coerce_audio_pauses(
    pauses: Iterable[AudioPause | dict[str, Any] | tuple[float, float]],
) -> tuple[AudioPause, ...]:
    result: list[AudioPause] = []
    for item in pauses:
        if isinstance(item, AudioPause):
            pause = item
        elif isinstance(item, dict):
            pause = AudioPause(float(item.get("start_sec", item.get("start", 0))), float(item.get("end_sec", item.get("end", 0))))
        else:
            start, end = item
            pause = AudioPause(float(start), float(end))
        if pause.end_sec > pause.start_sec:
            result.append(pause)
    return tuple(sorted(result, key=lambda pause: (pause.start_sec, pause.end_sec)))


def _audio_silence_threshold(audio, config: dict[str, Any]) -> float:
    explicit = config.get("source_subtitle_audio_silence_threshold_dbfs")
    if explicit not in {None, ""}:
        return _config_float(explicit, -36.0)
    if audio.dBFS == float("-inf"):
        return -45.0
    offset = _config_float(
        config.get("source_subtitle_audio_pause_threshold_offset_db"),
        DEFAULT_AUDIO_PAUSE_THRESHOLD_OFFSET_DB,
    )
    return max(-50.0, min(-30.0, float(audio.dBFS) - offset))


def _looks_word_level(word_counts: list[int]) -> bool:
    if not word_counts:
        return False
    avg = sum(word_counts) / len(word_counts)
    med = median(word_counts)
    one_word_ratio = sum(1 for count in word_counts if count <= 1) / len(word_counts)
    return avg <= 2.0 or med <= 1 or one_word_ratio >= 0.6


def _is_sentence_end(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(str(text)) if token.strip()]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _config_int(value: object, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _config_float(value: object, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _config_bool(value: object, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)
