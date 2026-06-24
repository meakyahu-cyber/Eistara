from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess

from eistara.core.media.validation import is_usable_media_file, remove_unusable_media_file


def test_is_usable_media_file_requires_existing_nonempty_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.mp3"
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")

    assert is_usable_media_file(missing, require_audio=True) is False
    assert is_usable_media_file(empty, require_audio=True) is False


def test_is_usable_media_file_accepts_ffprobe_audio(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"not-empty")

    def fake_run(*_args, **_kwargs):
        return CompletedProcess(
            args=("ffprobe",),
            returncode=0,
            stdout='{"format":{"duration":"1.25"},"streams":[{"codec_type":"audio","duration":"1.25"}]}',
            stderr="",
        )

    monkeypatch.setattr("eistara.core.media.validation.subprocess.run", fake_run)

    assert is_usable_media_file(audio, require_audio=True) is True


def test_is_usable_media_file_rejects_ffprobe_failure(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"not-empty")

    def fake_run(*_args, **_kwargs):
        return CompletedProcess(args=("ffprobe",), returncode=1, stdout="", stderr="bad media")

    monkeypatch.setattr("eistara.core.media.validation.subprocess.run", fake_run)

    assert is_usable_media_file(audio, require_audio=True) is False


def test_remove_unusable_media_file_deletes_corrupt_nonempty_file(monkeypatch, tmp_path: Path) -> None:
    audio = tmp_path / "audio.mp3"
    audio.write_bytes(b"not-empty")

    def fake_run(*_args, **_kwargs):
        return CompletedProcess(args=("ffprobe",), returncode=1, stdout="", stderr="bad media")

    monkeypatch.setattr("eistara.core.media.validation.subprocess.run", fake_run)

    remove_unusable_media_file(audio, require_audio=True)

    assert not audio.exists()
