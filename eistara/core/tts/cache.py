from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audio import has_audible_audio, has_positive_audio_duration
from .models import TtsRequest, TtsSettings


TTS_AUDIO_CACHE_KEYS = (
    "merge_micro_lines",
    "merge_micro_line_chars",
    "postprocess_audio",
    "trim_silence",
    "trim_silence_padding_ms",
    "trim_min_silence_len_ms",
    "trim_silence_threshold_offset_db",
    "trim_silence_threshold_min_dbfs",
    "peak_normalize_dbfs",
    "lowpass_hz",
)

INDEXTTS_CACHE_KEYS = (
    "prompt_audio_mode",
    "prompt_audio",
    "emo_mode",
    "emo_weight",
    "use_random",
    "max_text_tokens_per_segment",
    "do_sample",
    "top_p",
    "top_k",
    "temperature",
    "length_penalty",
    "num_beams",
    "repetition_penalty",
    "max_mel_tokens",
    "duration_control",
)

CUSTOM_TTS_CACHE_KEYS = (
    "mode",
    "python_callable",
    "command",
    "placeholder_audio",
)


def cache_meta_path(audio_path: str | os.PathLike[str]) -> Path:
    return Path(audio_path).with_suffix(Path(audio_path).suffix + ".cache.json")


def file_fingerprint(path: str | os.PathLike[str] | None) -> dict[str, Any] | None:
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return {"path": str(file_path), "exists": False}
    stat = file_path.stat()
    return {
        "path": str(file_path.resolve()),
        "exists": True,
        "mtime": round(stat.st_mtime, 3),
        "size": stat.st_size,
    }


@dataclass(slots=True)
class TtsCachePolicy:
    settings: TtsSettings

    def build_metadata(self, request: TtsRequest, cleaned_text: str) -> dict[str, Any]:
        method = self.settings.method
        payload = {
            "version": self.settings.cache_version,
            "text": cleaned_text,
            "speaker": request.speaker,
            "voice": request.voice,
            "tts_method": method,
            "tts_audio": _selected_config(self.settings.audio_config, TTS_AUDIO_CACHE_KEYS),
        }
        if method == "indextts":
            prompt_audio = self.settings.provider_config.get("prompt_audio")
            payload["indextts"] = {
                "effective_prompt_audio": prompt_audio or "",
                "effective_prompt_audio_file": file_fingerprint(prompt_audio),
                "config": _selected_config(self.settings.provider_config, INDEXTTS_CACHE_KEYS),
            }
            request_duration_control = _request_duration_control(request.metadata)
            if request_duration_control:
                payload["indextts"]["request_duration_control"] = request_duration_control
        elif method == "custom_tts":
            payload["custom_tts"] = {
                "config": _selected_config(self.settings.provider_config, CUSTOM_TTS_CACHE_KEYS),
            }
        else:
            payload["voice"] = request.voice
            payload["provider_config"] = self.settings.provider_config
            payload["request_metadata"] = request.metadata
        signature = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return {"signature": signature, "payload": payload}

    def read_metadata(self, audio_path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(cache_meta_path(audio_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def write_metadata(self, audio_path: Path, metadata: dict[str, Any], legacy_adopted: bool = False) -> None:
        output = dict(metadata)
        output["legacy_adopted"] = bool(legacy_adopted)
        meta_path = cache_meta_path(audio_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    def remove(self, audio_path: Path) -> None:
        for path in (audio_path, cache_meta_path(audio_path)):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def should_skip(self, audio_path: Path, metadata: dict[str, Any]) -> bool:
        if not audio_path.exists():
            return False
        if not has_audible_audio(audio_path):
            self.remove(audio_path)
            return False
        cached = self.read_metadata(audio_path)
        if cached and cached.get("signature") == metadata["signature"]:
            if has_positive_audio_duration(audio_path):
                return True
            self.remove(audio_path)
            return False
        if cached:
            self.remove(audio_path)
            return False
        if has_positive_audio_duration(audio_path):
            self.write_metadata(audio_path, metadata, legacy_adopted=True)
            return True
        return False


def _selected_config(config: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: config.get(key) for key in keys}


def _request_duration_control(metadata: dict[str, Any]) -> dict[str, Any]:
    for key in ("indextts_duration_control", "duration_control"):
        value = metadata.get(key)
        if isinstance(value, dict):
            return dict(value)
    return {}
