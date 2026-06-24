from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from eistara.core.media.validation import is_usable_media_file, remove_unusable_media_file


class DemucsAdapterError(RuntimeError):
    """Raised when Demucs vocal/background separation fails."""


class DemucsVocalSeparationProvider:
    name = "demucs"

    def separate(self, source_audio: str | Path, output_dir: str | Path, *, segment_minutes: float = 30.0) -> tuple[Path, Path]:
        return demucs_audio(source_audio, output_dir, segment_minutes=segment_minutes)

    def normalize(self, audio_path: str | Path, output_path: str | Path, *, format: str = "mp3") -> Path:
        return normalize_audio_volume(audio_path, output_path, format=format)


def demucs_audio(source_audio: str | Path, output_dir: str | Path, *, segment_minutes: float = 30.0) -> tuple[Path, Path]:
    """Separate vocals/background into Eistara output paths.

    Eistara writes ``output/audio/vocal.mp3`` and ``output/audio/background.mp3`` and
    skips work when both already exist.
    """

    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    vocal_path = audio_dir / "vocal.mp3"
    background_path = audio_dir / "background.mp3"
    if _is_usable_audio_file(vocal_path) and _is_usable_audio_file(background_path):
        return vocal_path, background_path
    _remove_unusable_audio_file(vocal_path)
    _remove_unusable_audio_file(background_path)

    try:
        import torch
        from demucs.api import Separator
        from demucs.audio import save_audio
        from demucs.pretrained import get_model
        from torch.cuda import is_available as is_cuda_available
    except ModuleNotFoundError as exc:
        if exc.name == "demucs.api":
            return _demucs_audio_cli(source_audio, audio_dir, vocal_path, background_path)
        raise DemucsAdapterError("demucs/torch packages are not available") from exc
    except Exception as exc:
        raise DemucsAdapterError("demucs/torch packages are not available") from exc

    source_audio = Path(source_audio)
    model = get_model("htdemucs")
    device = "cuda" if is_cuda_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    separator = Separator(
        model="htdemucs",
        device=device,
        shifts=1,
        overlap=0.25,
        split=True,
        segment=None,
        jobs=0,
        progress=False,
    )
    try:
        _, outputs = separator.separate_audio_file(str(source_audio))
        vocals = outputs.pop("vocals")
        _save_audio_mp3(save_audio, vocals.cpu(), vocal_path, samplerate=model.samplerate, bitrate=128)
        if not _is_usable_audio_file(vocal_path):
            raise DemucsAdapterError(f"Demucs vocal output is not a readable audio file: {vocal_path}")
        _release(torch)

        background = _mix_stems(stem.cpu() for stem in outputs.values())
        _save_audio_mp3(save_audio, background, background_path, samplerate=model.samplerate, bitrate=128)
        if not _is_usable_audio_file(background_path):
            raise DemucsAdapterError(f"Demucs background output is not a readable audio file: {background_path}")
        return vocal_path, background_path
    except Exception as exc:
        raise DemucsAdapterError(f"Demucs separation failed: {exc}") from exc
    finally:
        try:
            del separator, model
        except Exception:
            pass
        _release(torch)


def normalize_audio_volume(audio_path: str | Path, output_path: str | Path, *, target_db: float = -20.0, format: str = "mp3") -> Path:
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise DemucsAdapterError("pydub package is not available for vocal normalization") from exc

    audio_path = Path(audio_path)
    output_path = Path(output_path)
    audio = AudioSegment.from_file(audio_path)
    normalized = audio.apply_gain(target_db - audio.dBFS)
    normalized.export(output_path, format=format)
    return output_path


def _save_audio_mp3(save_audio, audio, path: Path, *, samplerate: int, bitrate: int) -> None:
    tmp_wav = path.with_name(f"{path.stem}.demucs_tmp.wav")
    try:
        save_audio(
            audio,
            tmp_wav,
            samplerate=samplerate,
            bitrate=bitrate,
            preset=2,
            clip="rescale",
            as_float=False,
            bits_per_sample=16,
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(tmp_wav),
                "-c:a",
                "libmp3lame",
                "-b:a",
                f"{int(bitrate)}k",
                str(path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            tmp_wav.unlink()
        except FileNotFoundError:
            pass


def _is_usable_audio_file(path: Path) -> bool:
    return is_usable_media_file(path, require_audio=True)


def _remove_unusable_audio_file(path: Path) -> None:
    remove_unusable_media_file(path, require_audio=True)


def _mix_stems(stems: Iterable):
    mixed = None
    for stem in stems:
        mixed = stem.clone() if mixed is None else mixed.add_(stem)
    if mixed is None:
        raise DemucsAdapterError("No Demucs background stems available")
    return mixed


def _release(torch_module) -> None:
    gc.collect()
    try:
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
    except Exception:
        pass


def _demucs_audio_cli(source_audio: str | Path, audio_dir: Path, vocal_path: Path, background_path: Path) -> tuple[Path, Path]:
    """Fallback for public demucs 4.0.x, which exposes CLI but not demucs.api."""
    work_dir = audio_dir / "_demucs_cli"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "demucs.separate",
        "-n",
        "htdemucs",
        "--two-stems",
        "vocals",
        "--mp3",
        "--mp3-bitrate",
        "128",
        "--mp3-preset",
        "2",
        "-j",
        "0",
        "--out",
        str(work_dir),
        str(source_audio),
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DemucsAdapterError(f"Demucs CLI failed with exit code {result.returncode}: {detail}")
        vocal_candidate = _find_demucs_cli_output(work_dir, "vocals.mp3")
        background_candidate = _find_demucs_cli_output(work_dir, "no_vocals.mp3")
        shutil.copy2(vocal_candidate, vocal_path)
        shutil.copy2(background_candidate, background_path)
        if not _is_usable_audio_file(vocal_path):
            raise DemucsAdapterError(f"Demucs vocal output is not a readable audio file: {vocal_path}")
        if not _is_usable_audio_file(background_path):
            raise DemucsAdapterError(f"Demucs background output is not a readable audio file: {background_path}")
        return vocal_path, background_path
    finally:
        try:
            shutil.rmtree(work_dir)
        except FileNotFoundError:
            pass


def _find_demucs_cli_output(root: Path, filename: str) -> Path:
    matches = sorted(path for path in root.rglob(filename) if path.is_file())
    if not matches:
        raise DemucsAdapterError(f"Demucs CLI did not produce {filename}")
    return matches[0]
