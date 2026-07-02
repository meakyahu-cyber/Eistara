from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from eistara.core.timeline import build_group_source_windows, segment_group_id
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


def build_indextts_payload(text: str, config: dict[str, Any], duration_control: dict[str, Any] | None = None) -> dict[str, Any]:
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
    duration_payload = duration_control or {}
    if duration_payload:
        payload.update(duration_payload)
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
        duration_control = _duration_control_from_request(request, settings.provider_config)
        payload = build_indextts_payload(request.text, settings.provider_config, duration_control=duration_control)
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

    def prepare_request_metadata(
        self,
        segment: dict[str, Any],
        metadata: dict[str, Any],
        settings: TtsSettings,
    ) -> dict[str, Any]:
        if _adaptive_source_window_duration_control_policy(settings.provider_config) is None:
            return metadata
        prepared = dict(metadata)
        prepared["indextts_duration_control"] = {"enabled": False}
        return prepared

    def build_retry_plan(
        self,
        segments: list[dict[str, Any]],
        durations: dict[str, float],
        settings: TtsSettings,
        *,
        source_duration_sec: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        policy = _adaptive_source_window_duration_control_policy(settings.provider_config)
        if policy is None:
            return {}
        return _adaptive_source_window_duration_control_targets(
            segments,
            durations,
            policy,
            source_duration_sec=source_duration_sec,
        )


def _duration_control_from_request(request: TtsRequest, config: dict[str, Any]) -> dict[str, Any]:
    metadata = request.metadata or {}
    request_config = metadata.get("indextts_duration_control") or metadata.get("duration_control")
    if isinstance(request_config, dict):
        return _duration_control_payload(request_config)
    return {}


def _duration_control_payload(config: dict[str, Any]) -> dict[str, Any]:
    enabled = _as_bool(config.get("enabled", config.get("duration_control_enabled")), False)
    if not enabled:
        return {}
    payload: dict[str, Any] = {"duration_control_enabled": True}
    target_duration_sec = _as_optional_float(config.get("target_duration_sec"))
    duration_scale = _as_optional_float(config.get("duration_scale"))
    if target_duration_sec is not None:
        payload["target_duration_sec"] = target_duration_sec
    if duration_scale is not None:
        payload["duration_scale"] = duration_scale
    return payload


def _adaptive_source_window_duration_control_policy(provider_config: dict[str, Any]) -> dict[str, float] | None:
    config = provider_config.get("duration_control")
    if not isinstance(config, dict):
        return None
    source_window_config = config.get("adaptive_source_window_retry")
    if not isinstance(source_window_config, dict) or not _as_bool(source_window_config.get("enabled"), False):
        return None
    return {
        "target_scale": max(0.05, _as_float(source_window_config.get("target_scale"), 1.0)),
        "min_target_sec": max(0.0, _as_float(source_window_config.get("min_target_sec"), 0.18)),
        "max_target_sec": _as_optional_float(source_window_config.get("max_target_sec")) or 0.0,
        "screen_borrow_max_sec": max(0.0, _as_float(source_window_config.get("screen_borrow_max_sec"), 0.30)),
        "screen_max_audio_speed": max(1.0, _as_float(source_window_config.get("screen_max_audio_speed"), 1.07)),
        "target_borrow_max_sec": max(0.0, _as_float(source_window_config.get("target_borrow_max_sec"), 0.50)),
        "borrow_min_seam_sec": max(0.0, _as_float(source_window_config.get("borrow_min_seam_sec"), 0.12)),
        "max_window_gap_sec": max(0.0, _as_float(source_window_config.get("max_window_gap_sec"), 6.0)),
        "line_gap_sec": max(0.0, _as_float(source_window_config.get("line_gap_sec"), 0.18)),
        "low_occupancy_retry_enabled": 1.0
        if _as_bool(source_window_config.get("low_occupancy_retry_enabled"), False)
        else 0.0,
        "low_occupancy_min_window_sec": max(0.0, _as_float(source_window_config.get("low_occupancy_min_window_sec"), 3.0)),
        "low_occupancy_min_slack_sec": max(0.0, _as_float(source_window_config.get("low_occupancy_min_slack_sec"), 1.0)),
        "low_occupancy_min_ratio": min(0.99, max(0.05, _as_float(source_window_config.get("low_occupancy_min_ratio"), 0.70))),
        "low_occupancy_target_ratio": min(1.0, max(0.05, _as_float(source_window_config.get("low_occupancy_target_ratio"), 0.78))),
        "low_occupancy_max_duration_factor": max(
            1.0,
            _as_float(source_window_config.get("low_occupancy_max_duration_factor"), 1.35),
        ),
        "low_occupancy_min_spoken_weight": max(
            0.0,
            _as_float(source_window_config.get("low_occupancy_min_spoken_weight"), 10.0),
        ),
        "low_occupancy_min_gain_sec": max(0.0, _as_float(source_window_config.get("low_occupancy_min_gain_sec"), 0.35)),
    }


def _adaptive_source_window_duration_control_targets(
    segments: list[dict[str, Any]],
    durations: dict[str, float],
    policy: dict[str, float],
    *,
    source_duration_sec: float | None = None,
) -> dict[str, dict[str, Any]]:
    min_target_sec = float(policy["min_target_sec"])
    max_target_sec = float(policy["max_target_sec"])
    screen_borrow_max_sec = float(policy["screen_borrow_max_sec"])
    screen_max_audio_speed = float(policy["screen_max_audio_speed"])
    target_borrow_max_sec = float(policy["target_borrow_max_sec"])
    seam_sec = float(policy["borrow_min_seam_sec"])
    max_gap_sec = float(policy["max_window_gap_sec"])
    line_gap_sec = float(policy["line_gap_sec"])
    low_occupancy_enabled = bool(float(policy.get("low_occupancy_retry_enabled", 0.0)) > 0.0)
    low_occupancy_min_window_sec = float(policy.get("low_occupancy_min_window_sec", 3.0))
    low_occupancy_min_slack_sec = float(policy.get("low_occupancy_min_slack_sec", 1.0))
    low_occupancy_min_ratio = float(policy.get("low_occupancy_min_ratio", 0.70))
    low_occupancy_target_ratio = float(policy.get("low_occupancy_target_ratio", 0.78))
    low_occupancy_max_duration_factor = float(policy.get("low_occupancy_max_duration_factor", 1.35))
    low_occupancy_min_spoken_weight = float(policy.get("low_occupancy_min_spoken_weight", 10.0))
    low_occupancy_min_gain_sec = float(policy.get("low_occupancy_min_gain_sec", 0.35))
    groups = _segments_by_source_window(segments)
    windows = _source_window_group_windows(groups, max_gap_sec, source_duration_sec=source_duration_sec)
    targets: dict[str, dict[str, Any]] = {}
    previous_effective_end: float | None = None
    previous_group_was_natural_overflow = False
    for group, group_segments in sorted(
        groups.items(),
        key=lambda item: (
            float(windows.get(item[0], {}).get("source_start_sec", 0.0)),
            float(windows.get(item[0], {}).get("source_end_sec", 0.0)),
            item[0],
        ),
    ):
        window = windows.get(group)
        if not window:
            continue
        valid_segments = [
            segment
            for segment in sorted(group_segments, key=lambda segment: str(segment.get("id") or ""))
            if str(segment.get("id") or "") in durations and str(segment.get("text") or "").strip()
        ]
        if not valid_segments:
            continue
        source_start = float(window["source_start_sec"])
        window_end = float(window["window_end_sec"])
        raw_total_duration = sum(float(durations[str(segment.get("id"))]) for segment in valid_segments)
        if len(valid_segments) > 1:
            raw_total_duration += line_gap_sec * (len(valid_segments) - 1)
        natural_overflow_sec = max(0.0, source_start + raw_total_duration - window_end)
        screen_borrow_sec = _borrowable_previous_gap(
            source_start,
            previous_effective_end,
            seam_sec=seam_sec,
            max_borrow_sec=screen_borrow_max_sec,
            wanted_sec=natural_overflow_sec,
            previous_group_was_natural_overflow=previous_group_was_natural_overflow,
        )
        screen_available_duration = max(0.001, window_end - (source_start - screen_borrow_sec))
        required_screen_speed = max(1.0, raw_total_duration / screen_available_duration)
        applied_screen_speed = min(screen_max_audio_speed, required_screen_speed)
        effective_group_end = source_start - screen_borrow_sec
        effective_group_end += sum(float(durations[str(segment.get("id"))]) / applied_screen_speed for segment in valid_segments)
        if len(valid_segments) > 1:
            effective_group_end += line_gap_sec * (len(valid_segments) - 1)
        if required_screen_speed <= screen_max_audio_speed + 0.001:
            if low_occupancy_enabled:
                low_targets = _low_occupancy_duration_control_targets(
                    valid_segments,
                    durations,
                    group=group,
                    source_start=source_start,
                    source_end=float(window["source_end_sec"]),
                    window_end=window_end,
                    raw_total_duration=raw_total_duration,
                    line_gap_sec=line_gap_sec,
                    min_window_sec=low_occupancy_min_window_sec,
                    min_slack_sec=low_occupancy_min_slack_sec,
                    min_ratio=low_occupancy_min_ratio,
                    target_ratio=low_occupancy_target_ratio,
                    max_duration_factor=low_occupancy_max_duration_factor,
                    min_spoken_weight=low_occupancy_min_spoken_weight,
                    min_gain_sec=low_occupancy_min_gain_sec,
                )
                targets.update(low_targets)
            previous_effective_end = effective_group_end
            previous_group_was_natural_overflow = natural_overflow_sec > 0.001
            continue

        target_borrow_sec = _borrowable_previous_gap(
            source_start,
            previous_effective_end,
            seam_sec=seam_sec,
            max_borrow_sec=target_borrow_max_sec,
            wanted_sec=natural_overflow_sec,
            previous_group_was_natural_overflow=previous_group_was_natural_overflow,
        )
        target_available = max(0.001, (window_end - (source_start - target_borrow_sec)) * float(policy["target_scale"]))
        if len(valid_segments) > 1:
            target_available = max(0.001, target_available - line_gap_sec * (len(valid_segments) - 1))
        if max_target_sec > 0:
            target_available = min(target_available, max_target_sec)
        weights = [_spoken_weight(segment) for segment in valid_segments]
        total_weight = sum(weights) or float(len(valid_segments))
        for segment, weight in zip(valid_segments, weights):
            segment_id = str(segment.get("id") or "")
            if not segment_id:
                continue
            share = target_available * (weight / total_weight)
            target_duration_sec = round(max(min_target_sec, share), 3)
            metadata = dict(segment.get("metadata") or {})
            metadata["indextts_duration_control"] = {
                "enabled": True,
                "target_duration_sec": target_duration_sec,
            }
            targets[segment_id] = {
                "segment": segment,
                "metadata": metadata,
                "retry_kind": "indextts_duration_control",
                "target_duration_sec": target_duration_sec,
                "natural_duration_sec": round(float(durations[segment_id]), 3),
                "source_window_group": group,
                "source_window_start_sec": round(source_start, 3),
                "source_window_end_sec": round(float(window["source_end_sec"]), 3),
                "source_window_owned_gap_after_sec": round(float(window.get("owned_gap_after_sec") or 0.0), 3),
                "screen_borrow_sec": round(screen_borrow_sec, 3),
                "screen_max_audio_speed": round(screen_max_audio_speed, 3),
                "required_screen_speed": round(required_screen_speed, 3),
                "target_borrow_sec": round(target_borrow_sec, 3),
            }
        previous_effective_end = source_start - target_borrow_sec + target_available
        if len(valid_segments) > 1:
            previous_effective_end += line_gap_sec * (len(valid_segments) - 1)
        previous_group_was_natural_overflow = natural_overflow_sec > 0.001
    return targets


def _low_occupancy_duration_control_targets(
    segments: list[dict[str, Any]],
    durations: dict[str, float],
    *,
    group: str,
    source_start: float,
    source_end: float,
    window_end: float,
    raw_total_duration: float,
    line_gap_sec: float,
    min_window_sec: float,
    min_slack_sec: float,
    min_ratio: float,
    target_ratio: float,
    max_duration_factor: float,
    min_spoken_weight: float,
    min_gain_sec: float,
) -> dict[str, dict[str, Any]]:
    available_duration = max(0.001, float(window_end) - float(source_start))
    source_duration = max(0.001, float(source_end) - float(source_start))
    slack_sec = available_duration - float(raw_total_duration)
    occupancy_ratio = float(raw_total_duration) / available_duration
    total_weight = sum(_spoken_weight(segment) for segment in segments)
    if source_duration < min_window_sec:
        return {}
    if slack_sec < min_slack_sec:
        return {}
    if occupancy_ratio >= min_ratio:
        return {}
    if total_weight < min_spoken_weight:
        return {}
    gap_duration = max(0.0, line_gap_sec * (len(segments) - 1))
    raw_audio_duration = max(0.001, raw_total_duration - gap_duration)
    target_total_duration = min(
        available_duration,
        max(raw_total_duration + min_gain_sec, available_duration * max(min_ratio, target_ratio)),
        raw_total_duration * max(1.0, max_duration_factor),
    )
    target_audio_duration = max(0.001, target_total_duration - gap_duration)
    scale = target_audio_duration / raw_audio_duration
    if scale <= 1.001:
        return {}

    targets: dict[str, dict[str, Any]] = {}
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        if not segment_id or segment_id not in durations:
            continue
        natural_duration = float(durations[segment_id])
        target_duration_sec = round(max(natural_duration + 0.001, natural_duration * scale), 3)
        if target_duration_sec < natural_duration + min_gain_sec / max(1, len(segments)):
            continue
        metadata = dict(segment.get("metadata") or {})
        metadata["indextts_duration_control"] = {
            "enabled": True,
            "target_duration_sec": target_duration_sec,
        }
        targets[segment_id] = {
            "segment": segment,
            "metadata": metadata,
            "retry_kind": "indextts_low_occupancy_duration_control",
            "target_duration_sec": target_duration_sec,
            "natural_duration_sec": round(natural_duration, 3),
            "min_accept_duration_sec": round(natural_duration + min(0.15, max(0.001, min_gain_sec / 2)), 3),
            "source_window_group": group,
            "source_window_start_sec": round(source_start, 3),
            "source_window_end_sec": round(source_end, 3),
            "source_window_available_sec": round(available_duration, 3),
            "source_window_slack_sec": round(slack_sec, 3),
            "source_window_occupancy_ratio": round(occupancy_ratio, 3),
            "low_occupancy_min_ratio": round(min_ratio, 3),
            "low_occupancy_target_ratio": round(target_ratio, 3),
            "low_occupancy_duration_scale": round(scale, 3),
        }
    return targets


def _borrowable_previous_gap(
    source_start: float,
    previous_effective_end: float | None,
    *,
    seam_sec: float,
    max_borrow_sec: float,
    wanted_sec: float,
    previous_group_was_natural_overflow: bool,
) -> float:
    if wanted_sec <= 0.001 or previous_effective_end is None or previous_group_was_natural_overflow:
        return 0.0
    available_gap = max(0.0, float(source_start) - float(previous_effective_end) - seam_sec)
    return min(max(0.0, wanted_sec), max(0.0, max_borrow_sec), available_gap)


def _source_window_group_windows(
    groups: dict[str, list[dict[str, Any]]],
    max_gap_after_sec: float,
    *,
    source_duration_sec: float | None = None,
) -> dict[str, dict[str, float]]:
    grouped_segments: list[dict[str, Any]] = []
    for group, group_segments in groups.items():
        for segment in group_segments:
            grouped = dict(segment)
            grouped["number"] = group
            grouped_segments.append(grouped)
    windows = build_group_source_windows(
        grouped_segments,
        max_gap_after_sec=max_gap_after_sec,
        source_duration_sec=source_duration_sec,
        group_key="number",
    )
    return {
        str(group): {
            "source_start_sec": window.source_start_sec,
            "source_end_sec": window.source_end_sec,
            "owned_gap_after_sec": window.owned_gap_after_sec,
            "window_end_sec": window.window_end_sec,
        }
        for group, window in windows.items()
    }


def _segments_by_source_window(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for index, segment in enumerate(segments, 1):
        segment_id = str(segment.get("id") or index)
        group = str(segment.get("number") or segment_group_id(segment_id))
        groups.setdefault(group, []).append(segment)
    return groups


def _spoken_weight(segment: dict[str, Any]) -> float:
    text = str(segment.get("text") or segment.get("target_text") or "")
    chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_words = len(re.findall(r"[A-Za-z0-9]+", text))
    return float(max(1, chinese_chars + ascii_words))


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


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default
