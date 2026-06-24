from __future__ import annotations

from pathlib import Path

from eistara.adapters.asr import audio_separator


def test_audio_separator_provider_writes_standard_outputs_and_cleans_temp(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "model.onnx").write_bytes(b"model")

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.output_dir = Path(kwargs["output_dir"])

        def load_model(self, model_filename):
            assert model_filename == "model.onnx"

        def separate(self, audio_file_path, custom_output_names=None):
            assert Path(audio_file_path) == source
            assert custom_output_names["Vocals"] == "vocal"
            vocal = self.output_dir / "vocal.wav"
            background = self.output_dir / "background.wav"
            vocal.write_bytes(b"vocal")
            background.write_bytes(b"background")
            return [vocal.name, background.name]

    def fake_convert(source_path, target_path, *, bitrate="128k"):
        target_path.write_bytes(source_path.read_bytes())

    monkeypatch.setattr("audio_separator.separator.Separator", FakeSeparator)
    monkeypatch.setattr(audio_separator, "_convert_to_mp3", fake_convert)
    monkeypatch.setattr(audio_separator, "_is_usable_audio_file", lambda path: path.exists() and path.stat().st_size > 0)

    provider = audio_separator.AudioSeparatorVocalSeparationProvider(model_filename="model.onnx", model_dir=model_dir)

    vocal, background = provider.separate(source, tmp_path / "output", segment_minutes=1)

    assert vocal == tmp_path / "output" / "audio" / "vocal.mp3"
    assert background == tmp_path / "output" / "audio" / "background.mp3"
    assert vocal.read_bytes() == b"vocal"
    assert background.read_bytes() == b"background"
    assert not (tmp_path / "output" / "audio" / "_audio_separator").exists()


def test_audio_separator_provider_requires_local_model(tmp_path: Path) -> None:
    provider = audio_separator.AudioSeparatorVocalSeparationProvider(model_filename="missing.onnx", model_dir=tmp_path)

    try:
        provider.separate(tmp_path / "source.wav", tmp_path / "output")
    except audio_separator.AudioSeparatorAdapterError as exc:
        assert "model is missing" in str(exc)
    else:
        raise AssertionError("missing audio-separator model should fail before network download")
