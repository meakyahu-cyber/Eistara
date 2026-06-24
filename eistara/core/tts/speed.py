from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def v1_slow_fast_speed_factor(text: str, duration_sec: float, audio_config: dict[str, Any]) -> float:
    if not _config_bool(audio_config, "slow_fast_segments", False):
        return 1.0
    if duration_sec < _config_float(audio_config, "slow_fast_min_duration_sec", 1.2):
        return 1.0

    units = _count_weighted_speech_units(text)
    if units <= 0:
        return 1.0

    units_per_sec = units / duration_sec
    trigger = _config_float(audio_config, "slow_fast_trigger_units_per_sec", 5.55)
    if units_per_sec <= trigger:
        return 1.0

    target = _config_float(audio_config, "slow_fast_target_units_per_sec", 5.2)
    max_duration_factor = _config_float(audio_config, "slow_fast_max_duration_factor", 1.10)
    if target <= 0 or max_duration_factor <= 1.0:
        return 1.0

    duration_factor = min(max_duration_factor, max(1.0, units_per_sec / target))
    return round(1.0 / duration_factor, 3)


def adjust_audio_speed_in_place(audio_path: str | Path, speed_factor: float, *, ffmpeg_path: str = "ffmpeg") -> None:
    if abs(float(speed_factor) - 1.0) < 0.001:
        return
    path = Path(audio_path)
    suffix = path.suffix or ".wav"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=path.parent)
    temp_path = Path(temp.name)
    temp.close()
    try:
        cmd = (
            ffmpeg_path,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-filter:a",
            _build_atempo_filter(speed_factor),
            str(temp_path),
        )
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_path.replace(path)
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _count_weighted_speech_units(text: str) -> float:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", str(text)))
    ascii_words = re.findall(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*", str(text))
    return cjk_count + len(ascii_words) * 1.5


def _build_atempo_filter(speed: float) -> str:
    factors: list[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_float(config: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default
