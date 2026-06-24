from __future__ import annotations

import sys
import types
from pathlib import Path

from eistara.adapters.asr.whisper import FasterWhisperAsrProvider, _result_from_whisper_mapping
from eistara.core.asr import AsrRequest, AsrSettings


def test_result_from_whisper_mapping() -> None:
    result = _result_from_whisper_mapping(
        {
            "language": "en",
            "segments": [
                {"id": 3, "start": 0.5, "end": 1.5, "text": " hello ", "words": [{"word": "hello"}]},
            ],
        }
    )

    assert result.language == "en"
    assert result.segments[0].id == 3
    assert result.segments[0].start_sec == 0.5
    assert result.segments[0].words == ({"word": "hello"},)


def test_faster_whisper_provider_passes_cache_options(monkeypatch, tmp_path: Path) -> None:
    calls = {}

    class FakeWhisperModel:
        def __init__(self, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs

        def transcribe(self, audio_path, **kwargs):
            calls["audio_path"] = audio_path
            calls["transcribe_kwargs"] = kwargs
            return iter(()), types.SimpleNamespace(language="en")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeWhisperModel))
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")

    result = FasterWhisperAsrProvider().transcribe(
        AsrRequest(audio_path=audio_path, language="en"),
        AsrSettings(
            model="large-v3",
            provider_config={
                "device": "cpu",
                "compute_type": "int8",
                "download_root": str(tmp_path / "hf-cache"),
                "local_files_only": "true",
            },
        ),
    )

    assert result.language == "en"
    assert calls["model_name"] == "large-v3"
    assert calls["kwargs"]["device"] == "cpu"
    assert calls["kwargs"]["compute_type"] == "int8"
    assert calls["kwargs"]["download_root"] == str(tmp_path / "hf-cache")
    assert calls["kwargs"]["local_files_only"] is True
