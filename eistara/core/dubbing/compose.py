from __future__ import annotations

from pathlib import Path


def prepare_v1_normalized_dub_audio(
    dub_audio: Path,
    output_dir: Path,
    *,
    enabled: bool,
    target_dbfs: float,
) -> tuple[Path, list[str], Path | None]:
    if not enabled:
        return dub_audio, [], None
    if not dub_audio.exists():
        return dub_audio, [], None
    try:
        from pydub import AudioSegment
    except Exception as exc:
        return dub_audio, [f"dub audio normalization skipped: pydub is not available: {exc}"], None

    try:
        audio = AudioSegment.from_file(dub_audio)
    except Exception as exc:
        return dub_audio, [f"dub audio normalization skipped: failed to read {dub_audio}: {exc}"], None
    if len(audio) <= 0 or audio.dBFS == float("-inf"):
        return dub_audio, [f"dub audio normalization skipped: silent or empty audio: {dub_audio}"], None

    normalized = audio.apply_gain(float(target_dbfs) - audio.dBFS)
    normalized_path = output_dir / "normalized_dub.wav"
    normalized.export(normalized_path, format="wav")
    return normalized_path, [], normalized_path


def video_speed_warnings(video_speed: float, warn_min: float, warn_max: float) -> list[str]:
    if abs(float(video_speed) - 1.0) <= 0.001:
        return []
    if float(video_speed) < float(warn_min) or float(video_speed) > float(warn_max):
        return [f"Video retime speed is {float(video_speed):.3f}x, outside preferred range {float(warn_min):.3f}-{float(warn_max):.3f}."]
    return []
