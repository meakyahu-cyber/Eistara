from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from eistara.core.tts import TtsProviderError, TtsRequest, TtsServiceError, TtsSettings


DEFAULT_INDEXTTS_API_URL = "http://127.0.0.1:8010/tts"


class HttpResponse(Protocol):
    status_code: int
    content: bytes
    text: str

    def raise_for_status(self) -> None:
        """Raise for non-success HTTP status."""


class HttpTransport(Protocol):
    def get(self, url: str, *, timeout: float) -> HttpResponse:
        """HTTP GET."""

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> HttpResponse:
        """HTTP POST."""


class RequestsHttpTransport:
    def __init__(self) -> None:
        import requests

        self._requests = requests

    def get(self, url: str, *, timeout: float) -> HttpResponse:
        return self._requests.get(url, timeout=timeout)

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> HttpResponse:
        return self._requests.post(url, json=json, timeout=timeout)


def indextts_root_url(api_url: str) -> str:
    parsed = urlsplit(api_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def build_indextts_payload(text: str, config: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "text": text,
        "prompt_audio": config.get("prompt_audio", ""),
        "emo_mode": config.get("emo_mode", 0),
        "emo_weight": config.get("emo_weight", 0.65),
        "use_random": config.get("use_random", False),
        "max_text_tokens_per_segment": config.get("max_text_tokens_per_segment", 120),
        "do_sample": config.get("do_sample", True),
        "top_p": config.get("top_p", 0.8),
        "top_k": config.get("top_k", 30),
        "temperature": config.get("temperature", 0.8),
        "length_penalty": config.get("length_penalty", 0.0),
        "num_beams": config.get("num_beams", 3),
        "repetition_penalty": config.get("repetition_penalty", 10.0),
        "max_mel_tokens": config.get("max_mel_tokens", 1500),
    }
    if not payload["prompt_audio"]:
        payload.pop("prompt_audio")
    return payload


def prepare_indextts_prompt_audio(config: dict[str, Any]) -> str:
    if not _is_auto_prompt_mode(config):
        return str(config.get("prompt_audio") or "")

    output_dir = Path(str(config.get("output_dir") or "output"))
    reference_dir = Path(str(config.get("reference_audio_dir") or output_dir / "audio" / "refers"))
    _ensure_reference_audio_segments(output_dir, reference_dir, config)
    output_file = reference_dir / "indextts_prompt.wav"
    report_file = output_dir / "log" / "indextts_prompt_audio.json"
    fallback_prompt = str(config.get("prompt_audio") or "")

    if _prompt_audio_fresh(output_file, report_file, reference_dir):
        return str(output_file)

    candidates = _load_reference_candidates(reference_dir, config)
    if not candidates:
        return fallback_prompt

    target_sec = _as_float(config.get("auto_prompt_target_sec"), 12.0)
    min_prompt_sec = _as_float(config.get("auto_prompt_min_prompt_sec"), 6.0)
    prompt, selected_candidates, strategy, min_required_ms = _build_prompt_from_candidates(candidates, config, target_sec, min_prompt_sec)
    if len(prompt) < min_required_ms:
        return fallback_prompt

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    prompt = _normalize_prompt_audio(prompt, config)
    prompt.export(output_file, format="wav")
    selected = [
        {
            "path": candidate["path"],
            "original_duration": candidate["original_duration"],
            "duration": candidate["duration"],
            "dbfs": candidate["dbfs"],
            "active_ratio": candidate["active_ratio"],
            "silence_ratio": candidate["silence_ratio"],
            "max_silence_ms": candidate["max_silence_ms"],
            "nonsilent_chunks": candidate["nonsilent_chunks"],
            "score": round(candidate["score"], 3),
        }
        for candidate in selected_candidates
    ]
    report_file.write_text(
        json.dumps(
            {
                "algorithm_version": 4,
                "mode": "auto_ref",
                "strategy": strategy,
                "prompt_audio": str(output_file),
                "duration": round(len(prompt) / 1000, 3),
                "selected": selected,
                "candidate_count": len(candidates),
                "fallback_prompt_audio": fallback_prompt,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(output_file)


def _ensure_reference_audio_segments(output_dir: Path, reference_dir: Path, config: dict[str, Any]) -> None:
    try:
        from eistara.core.tts.reference_audio import extract_reference_audio_segments
    except Exception:
        return
    extract_reference_audio_segments(
        output_dir,
        vocal_audio=config.get("vocal_audio") or output_dir / "audio" / "vocal.mp3",
        reference_audio_dir=reference_dir,
        tts_tasks=config.get("tts_tasks") or output_dir / "audio" / "tts_tasks.xlsx",
    )


@dataclass(slots=True)
class IndexTtsProvider:
    transport: HttpTransport | None = None
    name: str = "indextts"

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = RequestsHttpTransport()

    def check_ready(self, settings: TtsSettings, timeout: float = 5.0) -> None:
        api_url = str(settings.provider_config.get("api_url") or DEFAULT_INDEXTTS_API_URL)
        root_url = indextts_root_url(api_url)
        try:
            self.transport.get(root_url, timeout=timeout)
        except Exception as exc:
            raise TtsServiceError(f"IndexTTS service is not reachable: {api_url}. Details: {exc}") from exc

    def prepare_settings(
        self,
        settings: TtsSettings,
        *,
        output_dir: str | Path,
        reference_audio_dir: str | Path | None = None,
    ) -> TtsSettings:
        config = dict(settings.provider_config)
        config["output_dir"] = str(output_dir)
        if reference_audio_dir:
            config["reference_audio_dir"] = str(reference_audio_dir)
        prompt_audio = prepare_indextts_prompt_audio(config)
        if prompt_audio:
            config["prompt_audio"] = prompt_audio
        return TtsSettings(
            method=settings.method,
            cache_version=settings.cache_version,
            max_retries=settings.max_retries,
            service_backoff_base_sec=settings.service_backoff_base_sec,
            provider_config=config,
            audio_config=dict(settings.audio_config),
        )

    def synthesize(self, request: TtsRequest, settings: TtsSettings) -> None:
        api_url = str(settings.provider_config.get("api_url") or DEFAULT_INDEXTTS_API_URL)
        timeout = float(settings.provider_config.get("timeout_sec") or 300)
        payload = build_indextts_payload(request.text, settings.provider_config)
        try:
            response = self.transport.post(api_url, json=payload, timeout=timeout)
        except Exception as exc:
            raise TtsServiceError(f"IndexTTS connection failure: {exc}") from exc

        if response.status_code >= 500:
            detail = (response.text or "")[:200]
            raise TtsServiceError(f"IndexTTS {response.status_code} server error: {detail}")
        if response.status_code >= 400:
            detail = (response.text or "")[:200]
            raise TtsProviderError(f"IndexTTS {response.status_code} request error: {detail}")

        try:
            response.raise_for_status()
        except Exception as exc:
            raise TtsProviderError(f"IndexTTS request failed: {exc}") from exc

        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        Path(request.output_path).write_bytes(response.content)


def _is_auto_prompt_mode(config: dict[str, Any]) -> bool:
    return str(config.get("prompt_audio_mode") or "fixed").lower() in {"auto", "auto_ref", "source", "source_ref"}


def _prompt_audio_fresh(output_file: Path, report_file: Path, reference_dir: Path) -> bool:
    if not output_file.exists() or output_file.stat().st_size <= 1024:
        return False
    try:
        report = json.loads(report_file.read_text(encoding="utf-8"))
    except Exception:
        return False
    if report.get("algorithm_version") != 4:
        return False
    source_files = [
        path
        for path in reference_dir.glob("*.wav")
        if path.name != "indextts_prompt.wav"
    ]
    newest_source_mtime = max((path.stat().st_mtime for path in source_files), default=0)
    return output_file.stat().st_mtime >= newest_source_mtime


def _load_reference_candidates(reference_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent, detect_silence
    except Exception:
        return []

    target_sec = _as_float(config.get("auto_prompt_target_sec"), 12.0)
    min_segment_sec = _as_float(config.get("auto_prompt_min_segment_sec"), 3.0)
    max_segment_sec = _as_float(config.get("auto_prompt_max_segment_sec"), 16.0)
    min_active_ratio = _as_float(config.get("auto_prompt_min_active_ratio"), 0.58)
    min_dbfs = _as_float(config.get("auto_prompt_min_dbfs"), -38.0)
    max_internal_silence_ms = _as_int(config.get("auto_prompt_max_internal_silence_ms"), 900)
    max_silence_ratio = _as_float(config.get("auto_prompt_max_silence_ratio"), 0.38)
    max_nonsilent_chunks = _as_int(config.get("auto_prompt_max_nonsilent_chunks"), 7)

    candidates: list[dict[str, Any]] = []
    for path in sorted(reference_dir.glob("*.wav")):
        if path.name == "indextts_prompt.wav":
            continue
        try:
            audio = AudioSegment.from_file(path).set_channels(1)
        except Exception:
            continue
        original_duration_sec = len(audio) / 1000
        audio = _trim_prompt_edges(audio, config)
        audio = _trim_to_voice(audio, detect_nonsilent)
        duration_sec = len(audio) / 1000
        if duration_sec < min_segment_sec or audio.dBFS == float("-inf") or audio.dBFS < min_dbfs:
            continue
        if duration_sec > max_segment_sec:
            audio = audio[: int(max_segment_sec * 1000)]
            duration_sec = len(audio) / 1000
        active_ratio = _voice_activity_ratio(audio, detect_nonsilent)
        if active_ratio < min_active_ratio:
            continue
        stats = _silence_stats(audio, detect_silence, detect_nonsilent)
        if stats["max_silence_ms"] > max_internal_silence_ms:
            continue
        if stats["silence_ratio"] > max_silence_ratio:
            continue
        if stats["nonsilent_chunks"] > max_nonsilent_chunks:
            continue
        candidates.append(
            {
                "path": str(path),
                "original_duration": round(original_duration_sec, 3),
                "duration": round(duration_sec, 3),
                "dbfs": round(audio.dBFS, 3),
                "active_ratio": round(active_ratio, 3),
                "silence_ratio": round(stats["silence_ratio"], 3),
                "max_silence_ms": stats["max_silence_ms"],
                "nonsilent_chunks": stats["nonsilent_chunks"],
                "score": _candidate_score(duration_sec, audio.dBFS, target_sec, active_ratio, stats),
                "audio": audio,
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _trim_to_voice(audio, detect_nonsilent):
    if len(audio) == 0 or audio.dBFS == float("-inf"):
        return audio
    silence_thresh = max(audio.dBFS - 22, -45)
    ranges = detect_nonsilent(audio, min_silence_len=250, silence_thresh=silence_thresh)
    if not ranges:
        return audio
    start_ms = max(0, ranges[0][0] - 100)
    end_ms = min(len(audio), ranges[-1][1] + 100)
    return audio[start_ms:end_ms]


def _trim_prompt_edges(audio, config: dict[str, Any]):
    edge_ms = _as_int(config.get("auto_prompt_edge_trim_ms"), 700)
    min_segment_ms = int(_as_float(config.get("auto_prompt_min_segment_sec"), 3.0) * 1000)
    if edge_ms <= 0 or len(audio) <= min_segment_ms + edge_ms:
        return audio
    trim_ms = min(edge_ms, max(0, (len(audio) - min_segment_ms) // 2))
    return audio[trim_ms : len(audio) - trim_ms] if trim_ms > 0 else audio


def _voice_activity_ratio(audio, detect_nonsilent) -> float:
    if len(audio) == 0 or audio.dBFS == float("-inf"):
        return 0.0
    silence_thresh = max(audio.dBFS - 22, -45)
    ranges = detect_nonsilent(audio, min_silence_len=180, silence_thresh=silence_thresh)
    if not ranges:
        return 0.0
    active_ms = sum(end - start for start, end in ranges)
    return active_ms / len(audio)


def _silence_stats(audio, detect_silence, detect_nonsilent) -> dict[str, Any]:
    if len(audio) == 0 or audio.dBFS == float("-inf"):
        return {"silence_ratio": 1.0, "max_silence_ms": len(audio), "nonsilent_chunks": 0}
    silence_thresh = max(audio.dBFS - 22, -45)
    silence_ranges = detect_silence(audio, min_silence_len=180, silence_thresh=silence_thresh)
    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=180, silence_thresh=silence_thresh)
    silence_ms = sum(end - start for start, end in silence_ranges)
    return {
        "silence_ratio": silence_ms / len(audio),
        "max_silence_ms": max((end - start for start, end in silence_ranges), default=0),
        "nonsilent_chunks": len(nonsilent_ranges),
    }


def _candidate_score(duration_sec: float, dbfs: float, target_sec: float, active_ratio: float, stats: dict[str, Any]) -> float:
    duration_score = -abs(duration_sec - target_sec)
    loudness_score = -abs(dbfs + 22) * 0.08 if math.isfinite(dbfs) else -10
    activity_score = -abs(active_ratio - 0.82) * 4
    pause_score = -stats["silence_ratio"] * 5
    long_pause_score = -max(0, stats["max_silence_ms"] - 700) / 350
    chunk_score = -max(0, stats["nonsilent_chunks"] - 4) * 0.45
    return duration_score + loudness_score + activity_score + pause_score + long_pause_score + chunk_score


def _build_prompt_from_candidates(candidates: list[dict[str, Any]], config: dict[str, Any], target_sec: float, min_prompt_sec: float):
    try:
        from pydub import AudioSegment
    except Exception:
        raise RuntimeError("pydub is not available")

    strategy = str(config.get("auto_prompt_strategy") or "global_best").lower()
    target_ms = int(target_sec * 1000)
    min_prompt_ms = int(min_prompt_sec * 1000)
    soft_single_ms = int(_as_float(config.get("auto_prompt_soft_min_single_sec"), 5.0) * 1000)
    if strategy in {"global_best", "best", "single"}:
        for candidate in candidates:
            if len(candidate["audio"]) >= min_prompt_ms:
                return candidate["audio"][:target_ms], [candidate], "global_best", min_prompt_ms
        for candidate in candidates:
            if len(candidate["audio"]) >= soft_single_ms:
                return candidate["audio"][:target_ms], [candidate], "global_best_soft", soft_single_ms

    selected: list[dict[str, Any]] = []
    prompt = AudioSegment.empty()
    for candidate in candidates:
        if len(prompt) > 0:
            prompt += AudioSegment.silent(duration=120, frame_rate=candidate["audio"].frame_rate)
        remaining_ms = max(0, target_ms - len(prompt))
        if remaining_ms == 0:
            break
        prompt += candidate["audio"][:remaining_ms]
        selected.append(candidate)
        if len(prompt) >= min_prompt_ms:
            break
    return prompt, selected, "combined_fallback", min_prompt_ms


def _normalize_prompt_audio(audio, config: dict[str, Any]):
    target_dbfs = _as_float(config.get("auto_prompt_target_dbfs"), -20.0)
    if audio.dBFS != float("-inf"):
        audio = audio.apply_gain(target_dbfs - audio.dBFS)
    peak_dbfs = _as_float(config.get("auto_prompt_peak_dbfs"), -3.0)
    if audio.max_dBFS > peak_dbfs:
        audio = audio.apply_gain(peak_dbfs - audio.max_dBFS)
    return audio.set_channels(1).set_frame_rate(24000)


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
