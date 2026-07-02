from __future__ import annotations

import functools
import io
import json
import os
import random
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.asr import AsrProviderError, AsrRequest, AsrResult, AsrSettings

from .model_cache import resolve_faster_whisper_model
from .whisper import _result_from_whisper_mapping


@dataclass(slots=True)
class V1WhisperRuntimeAsrProvider:
    """V1-compatible Whisper runtime selector.

    V1 uses ``whisper.runtime`` rather than an independent ASR provider switch:
    ``local`` means WhisperX local, ``cloud`` means 302 WhisperX, and
    ``elevenlabs`` means ElevenLabs speech-to-text.
    """

    name: str = "whisperx"

    def transcribe(self, request: AsrRequest, settings: AsrSettings) -> AsrResult:
        default_runtime = {"whisperx-302": "cloud", "elevenlabs": "elevenlabs"}.get(self.name, "local")
        runtime = str(settings.provider_config.get("runtime") or default_runtime).strip().lower()
        if self.name == "whisperx-302":
            runtime = "cloud"
        elif self.name == "elevenlabs":
            runtime = "elevenlabs"
        raw_audio = Path(request.audio_path)
        vocal_audio = Path(settings.provider_config.get("vocal_audio_path") or raw_audio)
        segment_minutes = _float(settings.provider_config.get("demucs_segment_minutes"), 30.0)
        target_len = max(60.0, segment_minutes * 60.0)
        segments = split_audio(raw_audio, target_len=target_len, win=60.0)

        combined: dict[str, Any] = {"segments": []}
        language: str | None = None
        for start, end in segments:
            if runtime == "local":
                result = _transcribe_whisperx_local(raw_audio, vocal_audio, start, end, request, settings)
            elif runtime == "cloud":
                result = _transcribe_whisperx_302(vocal_audio, start, end, request, settings)
            elif runtime == "elevenlabs":
                result = _transcribe_elevenlabs(vocal_audio, start, end, request, settings)
            else:
                raise AsrProviderError(f"Unsupported whisper.runtime: {runtime}")
            language = result.get("language") or language
            combined["segments"].extend(result.get("segments") or [])
        parsed = _result_from_whisper_mapping(combined)
        return AsrResult(parsed.segments, language=language or parsed.language)


def split_audio(audio_file: str | Path, target_len: float = 30 * 60, win: float = 60) -> list[tuple[float, float]]:
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_silence
        from pydub.utils import mediainfo
    except Exception as exc:
        raise AsrProviderError("pydub package is not available for V1 audio splitting") from exc

    audio_file = Path(audio_file)
    audio = AudioSegment.from_file(audio_file)
    try:
        duration = float(mediainfo(str(audio_file))["duration"])
    except Exception:
        duration = len(audio) / 1000.0
    if duration <= target_len + win:
        return [(0.0, duration)]

    segments: list[tuple[float, float]] = []
    pos = 0.0
    safe_margin = 0.5
    while pos < duration:
        if duration - pos <= target_len:
            segments.append((pos, duration))
            break

        threshold = pos + target_len
        ws = int((threshold - win) * 1000)
        we = int((threshold + win) * 1000)
        silence_regions = detect_silence(audio[ws:we], min_silence_len=int(safe_margin * 1000), silence_thresh=-30)
        silence_regions = [(s / 1000 + (threshold - win), e / 1000 + (threshold - win)) for s, e in silence_regions]
        valid_regions = [
            (start, end)
            for start, end in silence_regions
            if (end - start) >= (safe_margin * 2) and threshold <= start + safe_margin <= threshold + win
        ]
        split_at = valid_regions[0][0] + safe_margin if valid_regions else threshold
        segments.append((pos, split_at))
        pos = split_at
    return segments


def _transcribe_whisperx_local(
    raw_audio: Path,
    vocal_audio: Path,
    start: float,
    end: float,
    request: AsrRequest,
    settings: AsrSettings,
) -> dict[str, Any]:
    try:
        import torch
        import whisperx
        from whisperx.audio import SAMPLE_RATE as whisperx_sr
        from whisperx.audio import load_audio as whisperx_load_audio
    except Exception as exc:
        raise AsrProviderError("whisperx/torch packages are not available") from exc

    _patch_torch_load(torch)
    model_dir = Path(str(settings.provider_config.get("model_dir") or settings.provider_config.get("download_root") or "./_model_cache"))
    language = request.language or settings.language or str(settings.provider_config.get("language") or "en")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = 16 if gpu_mem > 8 else 2
        compute_type = "float16" if torch.cuda.is_bf16_supported() else "int8"
    else:
        batch_size = 1
        compute_type = "int8"

    if language == "zh":
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        model_name = resolve_faster_whisper_model(
            model_name,
            model_dir,
            local_alias="Belle-whisper-large-v3-zh-punct-fasterwhisper",
        )
    else:
        model_name = settings.model or str(settings.provider_config.get("model") or "large-v3")
        model_name = resolve_faster_whisper_model(model_name, model_dir)

    hf_endpoint = check_hf_mirror()
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}
    asr_options = {"temperatures": [0], "initial_prompt": request.prompt or ""}
    whisper_language = None if "auto" in str(language) else language
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=whisper_language,
        vad_options=vad_options,
        asr_options=asr_options,
        download_root=str(model_dir),
    )
    try:
        raw_audio_segment = _load_audio_segment(whisperx_load_audio, whisperx_sr, raw_audio, start, end)
        vocal_audio_segment = _load_audio_segment(whisperx_load_audio, whisperx_sr, vocal_audio, start, end)
        result = model.transcribe(
            raw_audio_segment,
            batch_size=batch_size,
            print_progress=_bool(settings.provider_config.get("print_progress"), False),
        )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    detected_language = result.get("language")
    if detected_language == "zh" and language != "zh":
        raise AsrProviderError("Please specify the transcription language as zh and try again!")

    model_a, metadata = whisperx.load_align_model(language_code=detected_language or language, device=device)
    try:
        result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    finally:
        del model_a
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result["language"] = detected_language or language
    _offset_result_timestamps(result, start)
    return result


def _transcribe_whisperx_302(
    vocal_audio: Path,
    start: float,
    end: float,
    request: AsrRequest,
    settings: AsrSettings,
) -> dict[str, Any]:
    try:
        import librosa
        import requests
        import soundfile as sf
    except Exception as exc:
        raise AsrProviderError("requests/librosa/soundfile packages are not available for 302 ASR") from exc

    log_file = _asr_log_file(settings, f"whisperx302_{start}_{end}.json")
    if log_file.exists():
        return json.loads(log_file.read_text(encoding="utf-8"))

    language = request.language or settings.language or str(settings.provider_config.get("language") or "en")
    y, sr = librosa.load(str(vocal_audio), sr=16000)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, y[start_sample:end_sample], sr, format="WAV", subtype="PCM_16")
    audio_buffer.seek(0)

    response = requests.request(
        "POST",
        "https://api.302.ai/302/whisperx",
        headers={"Authorization": f"Bearer {settings.provider_config.get('whisperX_302_api_key') or ''}"},
        data={"processing_type": "align", "language": language, "output": "raw"},
        files=[("audio_input", ("audio_slice.wav", audio_buffer, "application/octet-stream"))],
    )
    response_json = response.json()
    _offset_result_timestamps(response_json, start)
    response_json.setdefault("language", language)
    _write_json(log_file, response_json)
    return response_json


def _transcribe_elevenlabs(
    vocal_audio: Path,
    start: float,
    end: float,
    request: AsrRequest,
    settings: AsrSettings,
) -> dict[str, Any]:
    try:
        import librosa
        import requests
        import soundfile as sf
    except Exception as exc:
        raise AsrProviderError("requests/librosa/soundfile packages are not available for ElevenLabs ASR") from exc

    log_file = _asr_log_file(settings, f"elevenlabs_transcribe_{start}_{end}.json")
    if log_file.exists():
        return json.loads(log_file.read_text(encoding="utf-8"))

    language = request.language or settings.language or str(settings.provider_config.get("language") or "en")
    y, sr = librosa.load(str(vocal_audio), sr=16000)
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        temp_path = Path(temp_file.name)
        sf.write(str(temp_path), y[start_sample:end_sample], sr, format="MP3")
    try:
        with temp_path.open("rb") as audio_file:
            response = requests.post(
                "https://api.elevenlabs.io/v1/speech-to-text",
                headers={"xi-api-key": str(settings.provider_config.get("elevenlabs_api_key") or "")},
                data={
                    "model_id": "scribe_v1",
                    "timestamps_granularity": "word",
                    "language_code": language,
                    "diarize": True,
                    "num_speakers": None,
                    "tag_audio_events": False,
                },
                files={"file": (temp_path.name, audio_file, "audio/mpeg")},
            )
        parsed = elev2whisper(response.json())
        _offset_result_timestamps(parsed, start)
        parsed.setdefault("language", language)
        _write_json(log_file, parsed)
        return parsed
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def elev2whisper(elev_json: dict[str, Any], word_level_timestamp: bool = False) -> dict[str, Any]:
    words = elev_json.get("words", [])
    if not words:
        return {"segments": []}

    segments: list[dict[str, Any]] = []
    seg = {"text": "", "start": words[0]["start"], "end": words[0]["end"], "speaker_id": words[0].get("speaker_id"), "words": []}
    for prev, nxt in zip(words, words[1:] + [None]):
        seg["text"] += prev.get("text", "")
        seg["end"] = prev["end"]
        if word_level_timestamp:
            seg["words"].append({"text": prev.get("text", ""), "start": prev["start"], "end": prev["end"]})
        if nxt is None or (nxt["start"] - prev["end"] > 1) or (nxt.get("speaker_id") != seg.get("speaker_id")):
            seg["text"] = str(seg["text"]).strip()
            if not word_level_timestamp:
                seg.pop("words", None)
            segments.append(seg)
            if nxt is not None:
                seg = {"text": "", "start": nxt["start"], "end": nxt["end"], "speaker_id": nxt.get("speaker_id"), "words": []}
    return {"segments": segments}


def check_hf_mirror() -> str | None:
    mirrors = {"Official": "huggingface.co", "Mirror": "hf-mirror.com"}
    fastest_url = "https://huggingface.co"
    best_time = float("inf")
    for domain in mirrors.values():
        cmd = ["ping", "-n", "1", "-w", "3000", domain] if os.name == "nt" else ["ping", "-c", "1", "-W", "3", domain]
        start = time.time()
        try:
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4, check=False)
        except Exception:
            continue
        response_time = time.time() - start
        if result.returncode == 0 and response_time < best_time:
            best_time = response_time
            fastest_url = f"https://{domain}"
    return fastest_url


def _patch_torch_load(torch_module: Any) -> None:
    original = torch_module.load
    if getattr(original, "_eistara_weights_patch", False):
        return

    @functools.wraps(original)
    def patched(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return original(*args, **kwargs)

    patched._eistara_weights_patch = True
    torch_module.load = patched


def _load_audio_segment(load_audio, sample_rate: int, audio_file: Path, start: float, end: float):
    full_audio = load_audio(str(audio_file), sr=sample_rate)
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    return full_audio[start_sample:end_sample]


def _offset_result_timestamps(result: dict[str, Any], start: float) -> None:
    for segment in result.get("segments") or []:
        if "start" in segment:
            segment["start"] += start
        if "end" in segment:
            segment["end"] += start
        for word in segment.get("words") or []:
            if "start" in word:
                word["start"] += start
            if "end" in word:
                word["end"] += start


def _asr_log_file(settings: AsrSettings, name: str) -> Path:
    output_dir = Path(str(settings.provider_config.get("output_dir") or "output"))
    path = output_dir / "log" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any, default: bool) -> bool:
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


WhisperRuntimeAsrProvider = V1WhisperRuntimeAsrProvider
