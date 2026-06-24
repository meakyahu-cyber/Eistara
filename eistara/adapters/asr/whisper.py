from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.asr import AsrProviderError, AsrRequest, AsrResult, AsrSegment, AsrSettings


@dataclass(slots=True)
class WhisperAsrProvider:
    name: str = "whisper"

    def transcribe(self, request: AsrRequest, settings: AsrSettings) -> AsrResult:
        try:
            import whisper
        except Exception as exc:
            raise AsrProviderError("whisper package is not available") from exc

        model_name = settings.model or str(settings.provider_config.get("model") or "base")
        model = whisper.load_model(model_name)
        options: dict[str, Any] = {}
        language = request.language or settings.language
        if language:
            options["language"] = language
        if request.prompt:
            options["initial_prompt"] = request.prompt
        raw = model.transcribe(str(Path(request.audio_path)), **options)
        return _result_from_whisper_mapping(raw)


@dataclass(slots=True)
class FasterWhisperAsrProvider:
    name: str = "faster-whisper"

    def transcribe(self, request: AsrRequest, settings: AsrSettings) -> AsrResult:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise AsrProviderError("faster-whisper package is not available") from exc

        model_name = settings.model or str(settings.provider_config.get("model") or "base")
        device = str(settings.provider_config.get("device") or "auto")
        compute_type = str(settings.provider_config.get("compute_type") or "default")
        model_kwargs: dict[str, Any] = {}
        download_root = str(
            settings.provider_config.get("download_root")
            or settings.provider_config.get("model_dir")
            or settings.provider_config.get("cache_dir")
            or ""
        ).strip()
        if download_root:
            model_kwargs["download_root"] = str(Path(download_root).expanduser())
        if _as_bool(settings.provider_config.get("local_files_only"), False):
            model_kwargs["local_files_only"] = True
        model = WhisperModel(model_name, device=device, compute_type=compute_type, **model_kwargs)
        segments, info = model.transcribe(
            str(Path(request.audio_path)),
            language=request.language or settings.language,
            initial_prompt=request.prompt or None,
        )
        return AsrResult(
            segments=tuple(
                AsrSegment(
                    id=index,
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text=str(segment.text or ""),
                )
                for index, segment in enumerate(segments, 1)
            ),
            language=getattr(info, "language", None),
        )


def _result_from_whisper_mapping(raw: dict[str, Any]) -> AsrResult:
    return AsrResult(
        segments=tuple(
            AsrSegment(
                id=int(segment.get("id", index)),
                start_sec=float(segment.get("start", 0)),
                end_sec=float(segment.get("end", 0)),
                text=str(segment.get("text") or ""),
                speaker=_speaker_id(segment.get("speaker", segment.get("speaker_id"))),
                words=tuple(segment.get("words") or ()),
            )
            for index, segment in enumerate(raw.get("segments") or [], 1)
        ),
        language=raw.get("language"),
    )


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


def _speaker_id(value: Any) -> str:
    text = str(value or "").strip()
    return text if text else "SPEAKER_00"
