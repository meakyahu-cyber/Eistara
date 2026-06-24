from __future__ import annotations

import logging
import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

from eistara.core.media.validation import is_usable_media_file, remove_unusable_media_file

from .demucs import normalize_audio_volume


DEFAULT_AUDIO_SEPARATOR_MODEL = "UVR-MDX-NET-Voc_FT.onnx"
DEFAULT_AUDIO_SEPARATOR_MODEL_DIR = Path("./models/audio-separator")
KNOWN_AUDIO_SEPARATOR_MODELS: dict[str, dict[str, Any]] = {
    DEFAULT_AUDIO_SEPARATOR_MODEL: {
        "size": 66762490,
        "md5": "d21dc03e4b9ef397b47231f483af6db8",
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
        "mirror_url": "https://gh.llkk.cc/https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx",
    }
}


class AudioSeparatorAdapterError(RuntimeError):
    """Raised when audio-separator vocal/background separation fails."""


class AudioSeparatorVocalSeparationProvider:
    name = "audio-separator"

    def __init__(
        self,
        *,
        model_filename: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
        model_dir: str | Path = DEFAULT_AUDIO_SEPARATOR_MODEL_DIR,
    ) -> None:
        self.model_filename = model_filename
        self.model_dir = Path(model_dir)

    def separate(self, source_audio: str | Path, output_dir: str | Path, *, segment_minutes: float = 30.0) -> tuple[Path, Path]:
        return separate_with_audio_separator(
            source_audio,
            output_dir,
            model_filename=self.model_filename,
            model_dir=self.model_dir,
            segment_minutes=segment_minutes,
        )

    def normalize(self, audio_path: str | Path, output_path: str | Path, *, format: str = "mp3") -> Path:
        return normalize_audio_volume(audio_path, output_path, format=format)


def separate_with_audio_separator(
    source_audio: str | Path,
    output_dir: str | Path,
    *,
    model_filename: str = DEFAULT_AUDIO_SEPARATOR_MODEL,
    model_dir: str | Path = DEFAULT_AUDIO_SEPARATOR_MODEL_DIR,
    segment_minutes: float = 30.0,
) -> tuple[Path, Path]:
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = audio_dir / "vocal.mp3"
    background_path = audio_dir / "background.mp3"
    if _is_usable_audio_file(vocal_path) and _is_usable_audio_file(background_path):
        return vocal_path, background_path
    _remove_unusable_audio_file(vocal_path)
    _remove_unusable_audio_file(background_path)

    model_dir = Path(model_dir)
    model_path = model_dir / model_filename
    status = audio_separator_model_status(model_filename=model_filename, model_dir=model_dir)
    if not status["exists"]:
        raise AudioSeparatorAdapterError(f"audio-separator model is missing: {model_path}. Download it before selecting the audio-separator backend.")
    if status["valid"] is False:
        raise AudioSeparatorAdapterError(f"audio-separator model failed integrity check: {model_path}")

    temp_dir = audio_dir / "_audio_separator"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        from audio_separator.separator import Separator
    except Exception as exc:
        raise AudioSeparatorAdapterError("audio-separator package is not available") from exc

    try:
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(model_dir),
            output_dir=str(temp_dir),
            output_format="WAV",
            chunk_duration=max(0.0, float(segment_minutes)) * 60.0 if segment_minutes else None,
        )
        separator.load_model(model_filename)
        outputs = separator.separate(
            str(source_audio),
            custom_output_names={
                "Vocals": "vocal",
                "Instrumental": "background",
                "Other": "background",
            },
        )
        output_paths = [_resolve_output_path(temp_dir, item) for item in outputs]
        vocal_wav = _find_stem(output_paths, ("vocal", "vocals"))
        background_wav = _find_stem(output_paths, ("background", "instrumental", "other"))

        _convert_to_mp3(vocal_wav, vocal_path)
        _convert_to_mp3(background_wav, background_path)
        if not _is_usable_audio_file(vocal_path):
            raise AudioSeparatorAdapterError(f"audio-separator vocal output is not readable: {vocal_path}")
        if not _is_usable_audio_file(background_path):
            raise AudioSeparatorAdapterError(f"audio-separator background output is not readable: {background_path}")
        return vocal_path, background_path
    except Exception as exc:
        if isinstance(exc, AudioSeparatorAdapterError):
            raise
        raise AudioSeparatorAdapterError(f"audio-separator separation failed: {exc}") from exc
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _resolve_output_path(output_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else output_dir / path


def audio_separator_model_status(*, model_filename: str = DEFAULT_AUDIO_SEPARATOR_MODEL, model_dir: str | Path = DEFAULT_AUDIO_SEPARATOR_MODEL_DIR) -> dict[str, Any]:
    model_dir = Path(model_dir)
    model_path = model_dir / model_filename
    expected = KNOWN_AUDIO_SEPARATOR_MODELS.get(model_filename) or {}
    exists = model_path.exists()
    size = model_path.stat().st_size if exists else 0
    expected_size = expected.get("size")
    expected_md5 = expected.get("md5")
    actual_md5 = ""
    valid: bool | None = None
    if exists and expected_size:
        valid = size == int(expected_size)
        if valid and expected_md5:
            actual_md5 = _md5(model_path)
            valid = actual_md5.lower() == str(expected_md5).lower()
    elif exists:
        valid = None
    return {
        "path": str(model_path),
        "exists": exists,
        "size": size,
        "expected_size": int(expected_size) if expected_size else None,
        "md5": actual_md5,
        "expected_md5": expected_md5 or "",
        "valid": valid,
        "url": expected.get("url", ""),
        "mirror_url": expected.get("mirror_url", ""),
    }


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_stem(paths: list[Path], names: tuple[str, ...]) -> Path:
    for path in paths:
        lowered = path.stem.lower()
        if any(name in lowered for name in names) and path.exists():
            return path
    raise AudioSeparatorAdapterError(f"audio-separator did not produce expected stem: {', '.join(names)}")


def _convert_to_mp3(source: Path, target: Path, *, bitrate: str = "128k") -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-c:a",
            "libmp3lame",
            "-b:a",
            bitrate,
            str(target),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _is_usable_audio_file(path: Path) -> bool:
    return is_usable_media_file(path, require_audio=True)


def _remove_unusable_audio_file(path: Path) -> None:
    remove_unusable_media_file(path, require_audio=True)
