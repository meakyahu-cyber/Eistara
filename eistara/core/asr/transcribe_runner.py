from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.jobs import StageName
from eistara.core.media import MediaProvider, build_audio_extract_plan
from eistara.core.media.validation import is_usable_media_file, remove_unusable_media_file
from eistara.core.pipeline import StageContext, StageResult, output_internal_path, resolve_output_dir
from eistara.core.subtitle.nlp_split import (
    DEFAULT_LANGUAGE_SPLIT_WITH_SPACE,
    DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE,
    generate_split_by_nlp,
)

from .models import AsrRequest, AsrSettings
from .providers import AsrProvider, VocalSeparationProvider
from .service import AsrService, asr_segments_to_subtitle_rows


@dataclass(slots=True)
class TranscribeStageRunner:
    asr_provider: AsrProvider
    media_provider: MediaProvider
    settings: AsrSettings = AsrSettings()
    vocal_separation_provider: VocalSeparationProvider | None = None
    stage: StageName = StageName.TRANSCRIBE

    def run(self, context: StageContext) -> StageResult:
        output_dir = resolve_output_dir(context)
        audio_path = context.task.get("audio_path") or context.task.get("source_audio") or context.artifacts.get("raw_audio")
        warnings: list[str] = []

        if not audio_path:
            source_video = context.task.get("source_video") or context.artifacts.get("source_video")
            if not source_video:
                return StageResult(status="skipped", skipped=True, warnings=["No source_video or audio_path for transcribe"])
            high_quality_audio = output_dir / "audio" / "raw_hq.wav"
            audio_path = output_dir / "audio" / "raw.mp3"
            if not is_usable_media_file(high_quality_audio, require_audio=True):
                remove_unusable_media_file(high_quality_audio, require_audio=True)
                extract_hq = self.media_provider.extract_audio(
                    build_audio_extract_plan(
                        source_video,
                        high_quality_audio,
                        sample_rate_hz=44100,
                        channels=2,
                        audio_codec="pcm_s16le",
                    )
                )
                if not extract_hq.ok:
                    raise RuntimeError(extract_hq.stderr or extract_hq.stdout or "high quality audio extraction failed")
            if not is_usable_media_file(audio_path, require_audio=True):
                remove_unusable_media_file(audio_path, require_audio=True)
                extract = self.media_provider.extract_audio(
                    build_audio_extract_plan(
                        source_video,
                        audio_path,
                        sample_rate_hz=16000,
                        channels=1,
                        audio_codec="libmp3lame",
                        audio_bitrate="32k",
                        metadata={"encoding": "UTF-8"},
                    )
                )
                if not extract.ok:
                    extract = self.media_provider.extract_audio(
                        build_audio_extract_plan(
                            source_video,
                            audio_path,
                            sample_rate_hz=16000,
                            channels=1,
                            audio_codec="pcm_s16le",
                            output_format="wav",
                        )
                    )
                if not extract.ok:
                    raise RuntimeError(extract.stderr or extract.stdout or "audio extraction failed")
        else:
            audio_path = Path(audio_path)

        vocal_audio, background_audio = _run_vocal_separation(
            self.settings,
            self.vocal_separation_provider,
            output_dir,
            Path(audio_path),
        )

        language = context.task.get("source_language") or self.settings.language
        settings = _settings_for_output(self.settings, output_dir, Path(audio_path), vocal_audio=vocal_audio)
        result = AsrService(self.asr_provider, settings).transcribe(
            AsrRequest(
                audio_path=Path(audio_path),
                language=str(language) if language else None,
                prompt=str(context.task.get("asr_prompt") or ""),
            )
        )
        warnings.extend(result.warnings)
        rows = asr_segments_to_subtitle_rows(result.segments)
        rows_payload = [
            {
                "start_sec": row.start_sec,
                "end_sec": row.end_sec,
                "source": row.source,
                "target": row.target,
                "speaker": row.speaker,
                "speaker_id": row.speaker,
            }
            for row in rows
        ]
        subtitle_rows_json = _write_subtitle_rows_json(output_dir, rows_payload)
        cleaned_chunks = output_dir / "log" / "cleaned_chunks.xlsx"
        _write_cleaned_chunks(cleaned_chunks, result.segments)
        split_by_nlp = _generate_split_by_nlp_safe(
            cleaned_chunks,
            output_dir,
            _nlp_language(str(language) if language else None, result.language),
            self.settings,
            warnings,
        )
        return StageResult(
            outputs={
                "language": result.language,
                "raw_audio": str(audio_path),
                "high_quality_audio": str(output_dir / "audio" / "raw_hq.wav"),
                **({"vocal_audio": str(vocal_audio)} if vocal_audio else {}),
                **({"background_audio": str(background_audio)} if background_audio else {}),
                "cleaned_chunks": str(cleaned_chunks),
                **({"split_by_nlp": str(split_by_nlp)} if split_by_nlp else {}),
                "segments": [segment.to_dict() for segment in result.segments],
                "subtitle_rows": rows_payload,
                "subtitle_rows_json": str(subtitle_rows_json),
            },
            warnings=warnings,
        )


def _run_vocal_separation(
    settings: AsrSettings,
    vocal_separation_provider: VocalSeparationProvider | None,
    output_dir: Path,
    audio_path: Path,
) -> tuple[Path | None, Path | None]:
    if not _as_bool(settings.provider_config.get("demucs"), False):
        return None, None
    if vocal_separation_provider is None:
        raise RuntimeError("Vocal separation is enabled but no provider is configured")
    try:
        demucs_source = output_dir / "audio" / "raw_hq.wav"
        if not demucs_source.exists():
            demucs_source = audio_path
        vocal_audio, background_audio = vocal_separation_provider.separate(
            demucs_source,
            output_dir,
            segment_minutes=_float_or_default(settings.provider_config.get("demucs_segment_minutes"), 30.0),
        )
        vocal_audio = vocal_separation_provider.normalize(vocal_audio, vocal_audio, format="mp3")
        return vocal_audio, background_audio
    except Exception as exc:
        raise RuntimeError(f"Vocal separation failed: {exc}") from exc


def _write_subtitle_rows_json(output_dir: Path, rows_payload: list[dict[str, Any]]) -> Path:
    subtitle_rows_json = output_internal_path(output_dir, "subtitle_rows.json")
    subtitle_rows_json.parent.mkdir(parents=True, exist_ok=True)
    subtitle_rows_json.write_text(json.dumps({"rows": rows_payload}, ensure_ascii=False, indent=2), encoding="utf-8")
    return subtitle_rows_json


def _settings_for_output(settings: AsrSettings, output_dir: Path, raw_audio: Path, *, vocal_audio: Path | None = None) -> AsrSettings:
    provider_config = dict(settings.provider_config)
    provider_config["output_dir"] = str(output_dir)
    existing_vocal = output_dir / "audio" / "vocal.mp3"
    selected_vocal = vocal_audio or (existing_vocal if is_usable_media_file(existing_vocal, require_audio=True) else raw_audio)
    provider_config.setdefault("vocal_audio_path", str(selected_vocal))
    return AsrSettings(language=settings.language, model=settings.model, provider_config=provider_config)


def _write_cleaned_chunks(path: Path, segments) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for segment in segments:
        if segment.words:
            rows.extend(_word_rows(segment))
        else:
            rows.append(
                {
                    "text": _quote_text(segment.text),
                    "start": segment.start_sec,
                    "end": segment.end_sec,
                    "speaker_id": segment.speaker,
                }
            )
    df = pd.DataFrame(rows, columns=["text", "start", "end", "speaker_id"])
    if not df.empty:
        df = df[df["text"].astype(str).str.len() > 0]
        df = df[df["text"].astype(str).str.len() <= 32]
    df.to_excel(path, index=False)


def _generate_split_by_nlp_safe(
    cleaned_chunks: str | Path,
    output_dir: Path,
    language: str | None,
    settings: AsrSettings,
    warnings: list[str],
) -> Path | None:
    try:
        result = generate_split_by_nlp(
            cleaned_chunks,
            output_dir,
            language=language,
            spacy_model_map=dict(settings.provider_config.get("spacy_model_map") or {}),
            language_split_with_space=tuple(settings.provider_config.get("language_split_with_space") or DEFAULT_LANGUAGE_SPLIT_WITH_SPACE),
            language_split_without_space=tuple(
                settings.provider_config.get("language_split_without_space") or DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE
            ),
        )
        warnings.extend(result.warnings)
        return result.split_by_nlp
    except Exception as exc:
        warnings.append(f"NLP split generation failed; continuing without split_by_nlp: {exc}")
        return None


def _nlp_language(config_language: str | None, detected_language: str | None) -> str | None:
    if config_language and config_language.lower() != "auto":
        return config_language
    return detected_language or config_language


def _word_rows(segment) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous_end = float(segment.start_sec)
    for word in segment.words:
        text = str(word.get("word") or word.get("text") or "").replace("»", "").replace("«", "").strip()
        if not text or len(text) > 30:
            continue
        start = _float_or_default(word.get("start"), previous_end)
        end = _float_or_default(word.get("end"), max(start, previous_end))
        previous_end = end
        rows.append(
            {
                "text": _quote_text(text),
                "start": start,
                "end": end,
                "speaker_id": word.get("speaker_id") or getattr(segment, "speaker", "SPEAKER_00"),
            }
        )
    return rows


def _quote_text(text: str) -> str:
    return f'"{str(text).strip()}"'


def _float_or_default(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_bool(value: Any, default: bool = False) -> bool:
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
