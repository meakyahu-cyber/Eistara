from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TtsAudioQualityReport:
    path: str
    original_duration_ms: int = 0
    processed_duration_ms: int = 0
    leading_silence_ms: int = 0
    trailing_silence_ms: int = 0
    trimmed_leading_ms: int = 0
    trimmed_trailing_ms: int = 0
    internal_silences: list[dict[str, int]] = field(default_factory=list)
    max_internal_silence_ms: int = 0
    internal_silence_ratio: float = 0.0
    detected_silence_ratio: float = 0.0
    peak_dbfs: float | None = None
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "original_duration_ms": self.original_duration_ms,
            "processed_duration_ms": self.processed_duration_ms,
            "leading_silence_ms": self.leading_silence_ms,
            "trailing_silence_ms": self.trailing_silence_ms,
            "trimmed_leading_ms": self.trimmed_leading_ms,
            "trimmed_trailing_ms": self.trimmed_trailing_ms,
            "internal_silences": list(self.internal_silences),
            "max_internal_silence_ms": self.max_internal_silence_ms,
            "internal_silence_ratio": self.internal_silence_ratio,
            "detected_silence_ratio": self.detected_silence_ratio,
            "peak_dbfs": self.peak_dbfs,
            "warnings": list(self.warnings),
            "skipped": self.skipped,
            "error": self.error,
        }


def postprocess_generated_tts_audio(path: str | Path, audio_config: dict[str, Any]) -> list[str]:
    return postprocess_generated_tts_audio_with_report(path, audio_config).warnings


def postprocess_generated_tts_audio_with_report(path: str | Path, audio_config: dict[str, Any]) -> TtsAudioQualityReport:
    if not _config_bool(audio_config, "postprocess_audio", True):
        return TtsAudioQualityReport(path=str(path), skipped=True)

    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent
    except Exception as exc:
        warning = f"TTS postprocess skipped: pydub is not available: {exc}"
        return TtsAudioQualityReport(path=str(path), warnings=[warning], skipped=True, error=str(exc))

    output_path = Path(path)
    try:
        audio = AudioSegment.from_file(output_path)
    except Exception as exc:
        warning = f"TTS postprocess skipped: {exc}"
        return TtsAudioQualityReport(path=str(output_path), warnings=[warning], skipped=True, error=str(exc))

    if len(audio) == 0 or audio.dBFS == float("-inf"):
        return _quality_report(output_path, audio, audio_config, original_duration_ms=len(audio))

    original_duration_ms = len(audio)
    trimmed_leading_ms = 0
    trimmed_trailing_ms = 0
    if _config_bool(audio_config, "trim_silence", True):
        min_silence_len = _config_int(audio_config, "trim_min_silence_len_ms", 180)
        threshold_offset = _config_float(audio_config, "trim_silence_threshold_offset_db", 22.0)
        threshold_floor = _config_float(audio_config, "trim_silence_threshold_min_dbfs", -45.0)
        padding_ms = _config_int(audio_config, "trim_silence_padding_ms", 120)
        silence_thresh = max(audio.dBFS - threshold_offset, threshold_floor)
        ranges = detect_nonsilent(
            audio,
            min_silence_len=max(1, min_silence_len),
            silence_thresh=silence_thresh,
        )
        if ranges:
            start_ms = max(0, ranges[0][0] - padding_ms)
            end_ms = min(len(audio), ranges[-1][1] + padding_ms)
            if end_ms > start_ms:
                trimmed_leading_ms = start_ms
                trimmed_trailing_ms = max(0, len(audio) - end_ms)
                audio = audio[start_ms:end_ms]

    peak_target = _config_float(audio_config, "peak_normalize_dbfs", -3.0)
    if audio.max_dBFS != float("-inf") and audio.max_dBFS > peak_target:
        audio = audio.apply_gain(peak_target - audio.max_dBFS)

    audio.export(output_path, format=output_path.suffix.lstrip(".") or "wav")
    return _quality_report(
        output_path,
        audio,
        audio_config,
        original_duration_ms=original_duration_ms,
        trimmed_leading_ms=trimmed_leading_ms,
        trimmed_trailing_ms=trimmed_trailing_ms,
    )


def analyze_generated_tts_audio(path: str | Path, audio_config: dict[str, Any]) -> TtsAudioQualityReport:
    try:
        from pydub import AudioSegment
    except Exception as exc:
        warning = f"TTS audio quality check skipped: pydub is not available: {exc}"
        return TtsAudioQualityReport(path=str(path), warnings=[warning], skipped=True, error=str(exc))

    output_path = Path(path)
    try:
        audio = AudioSegment.from_file(output_path)
    except Exception as exc:
        warning = f"TTS audio quality check skipped: {exc}"
        return TtsAudioQualityReport(path=str(output_path), warnings=[warning], skipped=True, error=str(exc))
    return _quality_report(output_path, audio, audio_config, original_duration_ms=len(audio))


def _quality_report(
    path: Path,
    audio: Any,
    audio_config: dict[str, Any],
    *,
    original_duration_ms: int,
    trimmed_leading_ms: int = 0,
    trimmed_trailing_ms: int = 0,
) -> TtsAudioQualityReport:
    leading_ms = 0
    trailing_ms = 0
    internal_silences: list[dict[str, int]] = []
    max_internal_ms = 0
    internal_ratio = 0.0
    detected_ratio = 0.0
    peak_dbfs = None if audio.max_dBFS == float("-inf") else round(float(audio.max_dBFS), 3)
    warnings: list[str] = []

    if len(audio) <= 0:
        warnings.append("TTS audio is empty")
    elif peak_dbfs is None:
        warnings.append("TTS audio is silent")

    if len(audio) > 0 and audio.dBFS != float("-inf"):
        try:
            from pydub.silence import detect_nonsilent, detect_silence
        except Exception as exc:
            warnings.append(f"TTS audio quality check skipped: pydub silence tools are not available: {exc}")
            return TtsAudioQualityReport(
                path=str(path),
                original_duration_ms=original_duration_ms,
                processed_duration_ms=len(audio),
                trimmed_leading_ms=trimmed_leading_ms,
                trimmed_trailing_ms=trimmed_trailing_ms,
                peak_dbfs=peak_dbfs,
                warnings=warnings,
                skipped=True,
                error=str(exc),
            )

        min_silence_len = _config_int(audio_config, "trim_min_silence_len_ms", 180)
        threshold_offset = _config_float(audio_config, "trim_silence_threshold_offset_db", 22.0)
        threshold_floor = _config_float(audio_config, "trim_silence_threshold_min_dbfs", -45.0)
        silence_thresh = max(audio.dBFS - threshold_offset, threshold_floor)
        nonsilent = detect_nonsilent(
            audio,
            min_silence_len=max(1, min_silence_len),
            silence_thresh=silence_thresh,
        )
        if nonsilent:
            leading_ms = max(0, int(nonsilent[0][0]))
            trailing_ms = max(0, int(len(audio) - nonsilent[-1][1]))
            internal_min_ms = _config_int(audio_config, "generated_internal_silence_min_ms", 500)
            for left, right in zip(nonsilent, nonsilent[1:]):
                start_ms = int(left[1])
                end_ms = int(right[0])
                duration_ms = max(0, end_ms - start_ms)
                if duration_ms >= internal_min_ms:
                    internal_silences.append({"start_ms": start_ms, "end_ms": end_ms, "duration_ms": duration_ms})
            max_internal_ms = max((item["duration_ms"] for item in internal_silences), default=0)
            internal_total_ms = sum(item["duration_ms"] for item in internal_silences)
            internal_ratio = round(internal_total_ms / len(audio), 4) if len(audio) else 0.0
        silence_ranges = detect_silence(
            audio,
            min_silence_len=max(1, min_silence_len),
            silence_thresh=silence_thresh,
        )
        silence_total_ms = sum(max(0, int(end - start)) for start, end in silence_ranges)
        detected_ratio = round(silence_total_ms / len(audio), 4) if len(audio) else 0.0

    max_internal_allowed = _config_int(audio_config, "generated_max_internal_silence_ms", 1200)
    max_internal_ratio = _config_float(audio_config, "generated_max_internal_silence_ratio", 0.45)
    if max_internal_allowed > 0 and max_internal_ms > max_internal_allowed:
        warnings.append(f"TTS audio has long internal silence: {max_internal_ms}ms > {max_internal_allowed}ms")
    if max_internal_ratio > 0 and internal_ratio > max_internal_ratio:
        warnings.append(f"TTS audio has high internal silence ratio: {internal_ratio:.2%} > {max_internal_ratio:.2%}")

    return TtsAudioQualityReport(
        path=str(path),
        original_duration_ms=original_duration_ms,
        processed_duration_ms=len(audio),
        leading_silence_ms=leading_ms,
        trailing_silence_ms=trailing_ms,
        trimmed_leading_ms=trimmed_leading_ms,
        trimmed_trailing_ms=trimmed_trailing_ms,
        internal_silences=internal_silences,
        max_internal_silence_ms=max_internal_ms,
        internal_silence_ratio=internal_ratio,
        detected_silence_ratio=detected_ratio,
        peak_dbfs=peak_dbfs,
        warnings=warnings,
    )


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


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default
