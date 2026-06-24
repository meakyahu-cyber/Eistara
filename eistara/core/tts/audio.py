from __future__ import annotations

import wave
from pathlib import Path

from eistara.core.media.validation import is_usable_media_file


def write_silence_wav(
    path: str | Path,
    *,
    duration_ms: int = 100,
    sample_rate_hz: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = max(1, round(sample_rate_hz * max(1, duration_ms) / 1000))
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(max(1, channels))
        handle.setsampwidth(max(1, sample_width))
        handle.setframerate(max(1, sample_rate_hz))
        handle.writeframes(b"\x00" * frame_count * max(1, channels) * max(1, sample_width))


def wav_duration_sec(path: str | Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            if sample_rate <= 0:
                return None
            return handle.getnframes() / sample_rate
    except (OSError, EOFError, wave.Error):
        return None


def has_positive_audio_duration(path: str | Path) -> bool:
    audio_path = Path(path)
    if not audio_path.exists() or audio_path.stat().st_size <= 0:
        return False
    duration = wav_duration_sec(audio_path)
    if duration is not None:
        return duration > 0
    if audio_path.suffix.lower() in {".wav", ".wave"}:
        return False
    return is_usable_media_file(audio_path, require_audio=True)
