from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd

from eistara.core.delivery import ArtifactRole, SubtitleDeliveryGenerator, SubtitleRow
from eistara.core.jobs.models import StageName
from eistara.core.pipeline import StageContext, StageResult, output_internal_path
from eistara.core.subtitle import SubtitleEvent, format_srt_timestamp, normalize_subtitle_text, parse_time_seconds, render_srt
from eistara.core.tts.segments import write_tts_segments_json

from .llm import LlmClient
from .models import Terminology, TranslationItem, TranslationSettings
from .pacing import count_chinese_chars, count_english_words, estimate_spoken_cost_units
from .service import PublishTranslationService
from .summary import generate_terminology_summary


@dataclass(slots=True)
class PublishTranslationStageRunner:
    llm: LlmClient
    settings: TranslationSettings = TranslationSettings()
    stage: StageName = StageName.TRANSLATE

    def run(self, context: StageContext) -> StageResult:
        items, rows = _load_translation_input(context)
        if not items:
            return StageResult(status="skipped", skipped=True, warnings=["No translation input in task or artifacts"])

        output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
        settings = _settings_for_context(self.settings, context)
        terminology_source, explicit_terminology = _terminology_source(context, output_dir)
        custom_terms_path = _custom_terms_path(context, output_dir)
        terminology_json: Path | None = None
        if _should_generate_terminology(
            settings,
            terminology_source,
            explicit_terminology,
            output_dir,
            items,
            custom_terms_path=custom_terms_path,
        ):
            _ensure_publish_source_lines(output_dir, items)
            summary_result = generate_terminology_summary(
                self.llm,
                items,
                settings,
                output_dir,
                custom_terms_path=custom_terms_path,
            )
            terminology = summary_result.terminology
            terminology_json = summary_result.path
        else:
            terminology = _load_terminology(terminology_source)
            terminology_json = _terminology_json_path(terminology_source)

        result = PublishTranslationService(self.llm, settings).translate(items, terminology)

        output_dir.mkdir(parents=True, exist_ok=True)
        translations_json = output_internal_path(output_dir, "translations.json")
        translations_json.parent.mkdir(parents=True, exist_ok=True)
        translation_rows = _build_translation_rows(items, result.translations)
        translations_json.write_text(
            json.dumps({"translations": translation_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        localization_report_json = None
        if result.localization_reports:
            localization_report_json = output_internal_path(output_dir, "localization_second_pass.json")
            localization_report_json.parent.mkdir(parents=True, exist_ok=True)
            localization_report_json.write_text(
                json.dumps({"batches": result.localization_reports}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        v1_outputs = _write_v1_publish_outputs(
            output_dir,
            items,
            result.translations,
            rows,
            settings,
            localization_reports=result.localization_reports,
        )

        tts_segments = _build_tts_segments(items, result.translations, output_dir)
        tts_segments_json = write_tts_segments_json(output_dir, tts_segments)
        outputs: dict[str, Any] = {
            "translations": result.translations,
            "translation_count": len(result.translations),
            "translations_json": str(translations_json),
            "tts_segments": tts_segments,
            "tts_segments_json": str(tts_segments_json),
            "tts_segments_count": len(tts_segments),
            **v1_outputs,
        }
        if terminology_json is not None and terminology_json.exists():
            outputs["terminology_json"] = str(terminology_json)
        if localization_report_json is not None:
            outputs["localization_second_pass_report"] = str(localization_report_json)
        subtitle_rows_json = context.task.get("subtitle_rows_json") or context.artifacts.get("subtitle_rows_json")
        if subtitle_rows_json:
            outputs["subtitle_rows_json"] = str(subtitle_rows_json)
        if rows is not None:
            outputs["subtitle_rows"] = [
                {
                    "start_sec": row.start_sec,
                    "end_sec": row.end_sec,
                    "source": row.source,
                    "target": result.translations.get(index, row.target),
                    "speaker": row.speaker,
                    "speaker_id": row.speaker,
                }
                for index, row in zip([item.id for item in items], rows)
            ]
        return StageResult(outputs=outputs, warnings=result.warnings)


def _load_translation_input(context: StageContext) -> tuple[list[TranslationItem], list[SubtitleRow] | None]:
    raw_items = context.task.get("translation_items") or context.artifacts.get("translation_items") or []
    if raw_items:
        return [_translation_item_from_mapping(dict(item), index) for index, item in enumerate(raw_items, 1)], None

    output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
    rows_json = context.task.get("subtitle_rows_json") or context.artifacts.get("subtitle_rows_json")
    if rows_json:
        rows = SubtitleDeliveryGenerator.from_config(context.config).load_rows_json(rows_json)
        return [_translation_item_from_subtitle_row(row, index) for index, row in enumerate(rows, 1)], rows

    raw_rows = context.task.get("subtitle_rows") or context.artifacts.get("subtitle_rows") or []
    if raw_rows:
        rows = [SubtitleRow.from_mapping(dict(item)) for item in raw_rows]
        return [_translation_item_from_subtitle_row(row, index) for index, row in enumerate(rows, 1)], rows

    explicit_source_lines_path = _first_existing_path(
        context.task.get("publish_source_lines"),
        context.artifacts.get("publish_source_lines"),
    )
    if explicit_source_lines_path:
        cleaned_chunks = _first_existing_path(
            context.task.get("cleaned_chunks"),
            context.artifacts.get("cleaned_chunks"),
            output_dir / "log" / "cleaned_chunks.xlsx",
        )
        return _translation_items_from_source_lines(explicit_source_lines_path, cleaned_chunks), None

    source_lines_path = _first_existing_path(
        output_dir / "log" / "publish_source_lines.txt",
        output_dir / "log" / "split_by_nlp.txt",
    )
    if source_lines_path:
        cleaned_chunks = _first_existing_path(
            context.task.get("cleaned_chunks"),
            context.artifacts.get("cleaned_chunks"),
            output_dir / "log" / "cleaned_chunks.xlsx",
        )
        items = _translation_items_from_source_lines(source_lines_path, cleaned_chunks)
        return items, None

    return [], None


def _translation_item_from_mapping(data: dict[str, Any], index: int) -> TranslationItem:
    start = data.get("start_sec", data.get("start", ""))
    end = data.get("end_sec", data.get("end", ""))
    return TranslationItem(
        id=int(data.get("id") or data.get("number") or index),
        source=str(data.get("source") or data.get("text") or data.get("Source") or ""),
        start=str(start if start is not None else ""),
        end=str(end if end is not None else ""),
        duration_sec=_duration_from_mapping(data, start, end),
        speaker=_speaker_id(data.get("speaker", data.get("speaker_id"))),
    )


def _translation_item_from_subtitle_row(row: SubtitleRow, index: int) -> TranslationItem:
    return TranslationItem(
        id=index,
        source=row.source,
        start=format_srt_timestamp(row.start_sec, row.end_sec).split(" --> ")[0],
        end=format_srt_timestamp(row.start_sec, row.end_sec).split(" --> ")[1],
        duration_sec=max(0.0, row.end_sec - row.start_sec),
        speaker=row.speaker,
    )


def _duration_from_mapping(data: dict[str, Any], start: Any, end: Any) -> float | None:
    if data.get("duration_sec") is not None:
        return float(data["duration_sec"])
    try:
        return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return None


def _load_terminology(data: Any) -> Terminology:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
    if not isinstance(data, dict):
        return Terminology()
    return Terminology(theme=str(data.get("theme") or ""), terms=tuple(data.get("terms") or ()))


def _settings_for_context(settings: TranslationSettings, context: StageContext) -> TranslationSettings:
    language = (
        context.task.get("source_language")
        or context.task.get("language")
        or context.artifacts.get("source_language")
        or context.artifacts.get("language")
    )
    if language and settings.source_language == "source language":
        return replace(settings, source_language=str(language))
    return settings


def _terminology_source(context: StageContext, output_dir: Path) -> tuple[Any, bool]:
    for key in ("terminology", "terminology_json"):
        value = context.task.get(key)
        if value:
            return value, True
    for key in ("terminology", "terminology_json"):
        value = context.artifacts.get(key)
        if value:
            return value, True
    path = output_dir / "log" / "terminology.json"
    if path.exists():
        return path, False
    return None, False


def _should_generate_terminology(
    settings: TranslationSettings,
    terminology_source: Any,
    explicit_terminology: bool,
    output_dir: Path,
    items: list[TranslationItem],
    custom_terms_path: Path | None = None,
) -> bool:
    if not settings.use_summary or explicit_terminology:
        return False
    if len(items) <= 1 and custom_terms_path is None:
        return False
    if terminology_source is not None:
        return False
    return not (output_dir / "log" / "terminology.json").exists()


def _terminology_json_path(data: Any) -> Path | None:
    if isinstance(data, (str, Path)):
        path = Path(data)
        if path.exists():
            return path
    return None


def _custom_terms_path(context: StageContext, output_dir: Path) -> Path | None:
    explicit = _first_existing_path(
        context.task.get("custom_terms"),
        context.task.get("custom_terms_path"),
        context.artifacts.get("custom_terms"),
        context.artifacts.get("custom_terms_path"),
    )
    if explicit:
        return explicit
    return _first_existing_path(
        output_dir.parent / "custom_terms.xlsx",
        context.job_dir / "custom_terms.xlsx",
        Path.cwd() / "custom_terms.xlsx",
        Path("custom_terms.xlsx"),
    )


def _ensure_publish_source_lines(output_dir: Path, items: list[TranslationItem]) -> Path:
    source_lines = output_dir / "log" / "publish_source_lines.txt"
    if source_lines.exists():
        return source_lines
    source_lines.parent.mkdir(parents=True, exist_ok=True)
    source_lines.write_text("\n".join(item.source for item in items if item.source).strip() + "\n", encoding="utf-8")
    return source_lines


def _build_translation_rows(items: list[TranslationItem], translations: dict[int, str]) -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "start": item.start,
            "end": item.end,
            "source": item.source,
            "speaker": item.speaker,
            "speaker_id": item.speaker,
            "text": translations.get(item.id, ""),
        }
        for item in items
    ]


def _build_tts_segments(items: list[TranslationItem], translations: dict[int, str], output_dir: Path) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for item in items:
        text = translations.get(item.id, "")
        if not text:
            continue
        segment_id = str(item.id)
        segments.append(
            {
                "id": segment_id,
                "start": item.start,
                "end": item.end,
                "source": item.source,
                "speaker": item.speaker,
                "speaker_id": item.speaker,
                "text": text,
                "target_text": text,
                "output_path": str(output_dir / "audio" / "tmp" / f"{segment_id}_0_temp.wav"),
                "metadata": {
                    "source": item.source,
                    "start": item.start,
                    "end": item.end,
                    "duration_sec": item.duration_sec,
                    "speaker": item.speaker,
                },
            }
        )
    return segments


def _write_v1_publish_outputs(
    output_dir: Path,
    items: list[TranslationItem],
    translations: dict[int, str],
    rows: list[SubtitleRow] | None,
    settings: TranslationSettings,
    localization_reports: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    log_dir = output_dir / "log"
    audio_dir = output_dir / "audio"
    log_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    source_lines = log_dir / "publish_source_lines.txt"
    source_lines.write_text("\n".join(item.source for item in items if item.source).strip() + "\n", encoding="utf-8")

    cleaned_translations = {item_id: _clean_translation_text(text) for item_id, text in translations.items()}
    excel_rows = [{"Source": item.source, "Translation": cleaned_translations.get(item.id, "")} for item in items]
    df = pd.DataFrame(excel_rows, columns=["Source", "Translation"])
    publish_translation = log_dir / "publish_translation.xlsx"
    publish_subtitles = log_dir / "publish_subtitles.xlsx"
    publish_audio_script = log_dir / "publish_audio_script.xlsx"
    for path in (publish_translation, publish_subtitles, publish_audio_script):
        df.to_excel(path, index=False)

    subtitle_rows = _apply_v1_short_gap_extension(_translated_subtitle_rows(items, cleaned_translations, rows))
    written = SubtitleDeliveryGenerator.from_config(settings.raw_config).write_source_timeline_subtitles(subtitle_rows, output_dir)
    source_srt = written.get(ArtifactRole.SOURCE_SUBTITLE, output_dir / "src.srt")
    translated_srt = written.get(ArtifactRole.TARGET_SUBTITLE, output_dir / "trans.srt")
    audio_source_srt = audio_dir / "src_subs_for_audio.srt"
    audio_translated_srt = audio_dir / "trans_subs_for_audio.srt"
    _write_v1_audio_subtitles(subtitle_rows, audio_source_srt, audio_translated_srt)

    report_path = log_dir / "publish_translate_report.json"
    batches = _report_batches(items, settings, cleaned_translations)
    summary = _report_summary(items, batches)
    report_path.write_text(
        json.dumps(
            {
                "mode": "publish_fast",
                "source_items": len(items),
                "summary": summary,
                "batches": batches,
                "artifacts": {
                    "translation": str(publish_translation),
                    "subtitles": str(publish_subtitles),
                    "audio_script": str(publish_audio_script),
                    "source_srt": str(source_srt),
                    "translated_srt": str(translated_srt),
                    "audio_source_srt": str(audio_source_srt),
                    "audio_translated_srt": str(audio_translated_srt),
                },
                "localization_second_pass": localization_reports or [],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "translation": str(publish_translation),
        "subtitles": str(publish_subtitles),
        "audio_script": str(publish_audio_script),
        "publish_source_lines": str(source_lines),
        "source_srt": str(source_srt),
        "translated_srt": str(translated_srt),
        "audio_source_srt": str(audio_source_srt),
        "audio_translated_srt": str(audio_translated_srt),
        "publish_translate_report": str(report_path),
    }


def _translated_subtitle_rows(
    items: list[TranslationItem],
    translations: dict[int, str],
    rows: list[SubtitleRow] | None,
) -> list[SubtitleRow]:
    if rows is not None:
        return [
            SubtitleRow(
                start_sec=row.start_sec,
                end_sec=row.end_sec,
                source=row.source,
                target=translations.get(item.id, row.target),
                speaker=row.speaker,
            )
            for item, row in zip(items, rows)
        ]
    return [
        SubtitleRow(
            start_sec=parse_time_seconds(item.start),
            end_sec=parse_time_seconds(item.end),
            source=item.source,
            target=translations.get(item.id, ""),
            speaker=item.speaker,
        )
        for item in items
    ]


def _apply_v1_short_gap_extension(rows: list[SubtitleRow]) -> list[SubtitleRow]:
    adjusted = list(rows)
    for index in range(len(adjusted) - 1):
        current = adjusted[index]
        next_row = adjusted[index + 1]
        delta = next_row.start_sec - current.end_sec
        if 0 < delta < 1:
            adjusted[index] = SubtitleRow(
                start_sec=current.start_sec,
                end_sec=next_row.start_sec,
                source=current.source,
                target=current.target,
                speaker=current.speaker,
            )
    return adjusted


def _write_v1_audio_subtitles(rows: list[SubtitleRow], source_srt: Path, translated_srt: Path) -> None:
    source_srt.parent.mkdir(parents=True, exist_ok=True)
    source_srt.write_text(
        render_srt(
            [
                SubtitleEvent(row.start_sec, row.end_sec, (normalize_subtitle_text(row.source),))
                for row in rows
                if normalize_subtitle_text(row.source)
            ]
        ),
        encoding="utf-8",
    )
    translated_srt.write_text(
        render_srt(
            [
                SubtitleEvent(row.start_sec, row.end_sec, (normalize_subtitle_text(row.target),))
                for row in rows
                if normalize_subtitle_text(row.target)
            ]
        ),
        encoding="utf-8",
    )


def _clean_translation_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    cleaned = str(value).strip().strip('"')
    try:
        import autocorrect_py as autocorrect
    except Exception:
        return cleaned
    return autocorrect.format(cleaned)


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
    return None


def _translation_items_from_source_lines(source_lines_path: Path, cleaned_chunks: Path | None) -> list[TranslationItem]:
    lines = [line.strip() for line in source_lines_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    timestamps = _source_line_timestamps(lines, cleaned_chunks)
    items: list[TranslationItem] = []
    for index, source in enumerate(lines, 1):
        start_sec, end_sec = timestamps[index - 1]
        if start_sec is None or end_sec is None:
            start_text = ""
            end_text = ""
            duration = None
        else:
            timestamp = format_srt_timestamp(start_sec, end_sec)
            start_text, end_text = timestamp.split(" --> ")
            duration = round(max(0.0, end_sec - start_sec), 3)
        items.append(
            TranslationItem(
                id=index,
                source=source,
                start=start_text,
                end=end_text,
                duration_sec=duration,
                speaker="SPEAKER_00",
            )
        )
    return items


def _speaker_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "SPEAKER_00"


def _source_line_timestamps(lines: list[str], cleaned_chunks: Path | None) -> list[tuple[float | None, float | None]]:
    if not lines or cleaned_chunks is None or not cleaned_chunks.exists():
        return [(None, None)] * len(lines)
    try:
        df_words = pd.read_excel(cleaned_chunks)
        df_words["text"] = df_words["text"].astype(str).str.strip('"').str.strip()
        df_source = pd.DataFrame({"Source": lines})
        return _get_sentence_timestamps(df_words, df_source)
    except Exception as exc:
        print(f"Publish timestamp hint generation failed; continuing without hints: {exc}")
        return [(None, None)] * len(lines)


def _get_sentence_timestamps(df_words: pd.DataFrame, df_sentences: pd.DataFrame) -> list[tuple[float, float]]:
    time_stamp_list: list[tuple[float, float]] = []
    full_words_str = ""
    position_to_word_idx: dict[int, int] = {}

    for idx, word in enumerate(df_words["text"]):
        clean_word = _remove_punctuation(str(word).lower())
        start_pos = len(full_words_str)
        full_words_str += clean_word
        for pos in range(start_pos, len(full_words_str)):
            position_to_word_idx[pos] = idx

    current_pos = 0
    for _, sentence in df_sentences["Source"].items():
        clean_sentence = _remove_punctuation(str(sentence).lower()).replace(" ", "")
        sentence_len = len(clean_sentence)

        match_found = False
        while current_pos <= len(full_words_str) - sentence_len:
            if full_words_str[current_pos : current_pos + sentence_len] == clean_sentence:
                start_word_idx = position_to_word_idx[current_pos]
                end_word_idx = position_to_word_idx[current_pos + sentence_len - 1]
                time_stamp_list.append(
                    (
                        float(df_words["start"][start_word_idx]),
                        float(df_words["end"][end_word_idx]),
                    )
                )
                current_pos += sentence_len
                match_found = True
                break
            current_pos += 1

        if not match_found:
            raise ValueError(f"No exact match found for sentence: {sentence}")

    return time_stamp_list


def _remove_punctuation(text: str) -> str:
    import re

    text = re.sub(r"\s+", " ", str(text))
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()


def _report_batches(
    items: list[TranslationItem],
    settings: TranslationSettings,
    translations: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    batches: list[dict[str, Any]] = []
    max_lines = max(1, int(settings.max_batch_lines))
    max_chars = max(200, int(settings.max_batch_chars))
    current: list[TranslationItem] = []
    current_chars = 0
    for item in items:
        item_chars = len(str(item.source))
        if current and (len(current) >= max_lines or current_chars + item_chars > max_chars):
            batches.append(_report_batch(len(batches) + 1, current, translations))
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        batches.append(_report_batch(len(batches) + 1, current, translations))
    return batches


def _report_summary(items: list[TranslationItem], batches: list[dict[str, Any]]) -> dict[str, Any]:
    source_items = len(items) if items else sum(int(batch.get("items") or 0) for batch in batches)
    actual_zh_chars = _sum_numeric(batches, "actual_zh_chars")
    actual_spoken_cost = _sum_numeric(batches, "actual_spoken_cost")
    source_duration_sec = _sum_numeric(batches, "source_duration_sec")

    return {
        "batch_count": len(batches),
        "totals": {
            "source_items": source_items,
            "source_chars": sum(int(batch.get("source_chars") or 0) for batch in batches),
            "english_words": sum(int(batch.get("english_words") or 0) for batch in batches),
            "source_duration_sec": _round_optional(source_duration_sec),
            "actual_zh_chars": actual_zh_chars,
            "actual_spoken_cost": actual_spoken_cost,
        },
        "spoken_cost": {
            "actual_zh_chars_per_sec": _numeric_stats(batches, "actual_zh_chars_per_sec"),
            "actual_spoken_cost_per_sec": _numeric_stats(batches, "actual_spoken_cost_per_sec"),
        },
    }


def _numeric_stats(items: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    values = [_as_float(item.get(key)) for item in items]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return {"min": None, "max": None, "avg": None}
    return {
        "min": round(min(numbers), 3),
        "max": round(max(numbers), 3),
        "avg": round(sum(numbers) / len(numbers), 3),
    }


def _sum_numeric(items: list[dict[str, Any]], key: str) -> float | int | None:
    values = [_as_float(item.get(key)) for item in items]
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    total = sum(numbers)
    return int(total) if total.is_integer() else round(total, 3)


def _weighted_average(items: list[dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    pairs = [
        (value, weight)
        for value, weight in ((_as_float(item.get(value_key)), _as_float(item.get(weight_key))) for item in items)
        if value is not None and weight is not None and weight > 0
    ]
    if not pairs:
        return None
    total_weight = sum(weight for _, weight in pairs)
    if total_weight <= 0:
        return None
    return sum(value * weight for value, weight in pairs) / total_weight


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    value_float = float(value)
    return int(value_float) if value_float.is_integer() else round(value_float, 3)


def _report_batch(
    index: int,
    batch: list[TranslationItem],
    translations: dict[int, str] | None = None,
) -> dict[str, Any]:
    english_words = sum(count_english_words(item.source) for item in batch)
    source_duration_sec = sum(max(0.0, float(item.duration_sec or 0.0)) for item in batch)
    translated_zh_chars = (
        sum(count_chinese_chars((translations or {}).get(item.id, "")) for item in batch)
        if translations is not None
        else None
    )
    translated_spoken_cost = (
        sum(estimate_spoken_cost_units((translations or {}).get(item.id, "")) for item in batch)
        if translations is not None
        else None
    )
    return {
        "index": index,
        "items": len(batch),
        "first_id": batch[0].id,
        "last_id": batch[-1].id,
        "source_chars": sum(len(str(item.source)) for item in batch),
        "english_words": english_words,
        "source_duration_sec": round(source_duration_sec, 3),
        "actual_zh_chars": translated_zh_chars,
        "actual_spoken_cost": translated_spoken_cost,
        "actual_zh_chars_per_sec": (
            round(translated_zh_chars / source_duration_sec, 3)
            if translated_zh_chars is not None and source_duration_sec > 0
            else None
        ),
        "actual_spoken_cost_per_sec": (
            round(translated_spoken_cost / source_duration_sec, 3)
            if translated_spoken_cost is not None and source_duration_sec > 0
            else None
        ),
    }
