from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from eistara.core.asr import (
    AsrRequest,
    AsrSegment,
    AsrSettings,
    AsrService,
    AsrStageRunner,
    ScriptedAsrProvider,
    TranscribeStageRunner,
    asr_segments_to_subtitle_rows,
    normalize_asr_segments,
)
from eistara.core.asr.source_subtitles import AudioPause, detect_audio_pauses, normalize_source_subtitle_rows
from eistara.core.delivery import SubtitleRow
from eistara.core.media import MediaCommandResult
from eistara.core.jobs import StageName
from eistara.core.pipeline import StageContext


class FakeMediaProvider:
    name = "fake-media"

    def __init__(self, result: MediaCommandResult | None = None):
        self.result = result or MediaCommandResult(("extract",), 0)
        self.extract_calls = []

    def probe(self, path: str):
        raise NotImplementedError

    def extract_audio(self, plan):
        self.extract_calls.append(plan)
        plan.output_audio.parent.mkdir(parents=True, exist_ok=True)
        plan.output_audio.write_bytes(b"audio")
        return self.result

    def compose_video(self, plan):
        raise NotImplementedError


class FakeVocalSeparationProvider:
    name = "fake-demucs"

    def __init__(self):
        self.separate_calls = []
        self.normalize_calls = []

    def separate(self, source_audio, output_dir, *, segment_minutes):
        self.separate_calls.append((Path(source_audio), Path(output_dir), segment_minutes))
        audio_dir = Path(output_dir) / "audio"
        vocal = audio_dir / "vocal.mp3"
        background = audio_dir / "background.mp3"
        vocal.write_bytes(b"vocal")
        background.write_bytes(b"background")
        return vocal, background

    def normalize(self, audio_path, output_path, *, format):
        self.normalize_calls.append((Path(audio_path), Path(output_path), format))
        return Path(output_path)


class CapturingAsrProvider(ScriptedAsrProvider):
    def __init__(self, segments: list[AsrSegment] | None = None, language: str | None = "en"):
        super().__init__(segments, language=language)
        self.settings_calls: list[AsrSettings] = []

    def transcribe(self, request: AsrRequest, settings: AsrSettings):
        self.settings_calls.append(settings)
        return super().transcribe(request, settings)


def write_tone_silence_wav(path: Path, parts: list[tuple[str, float]], sample_rate: int = 16000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for kind, duration_sec in parts:
            frame_count = int(sample_rate * duration_sec)
            for index in range(frame_count):
                if kind == "tone":
                    value = int(10000 * math.sin(2 * math.pi * 440 * index / sample_rate))
                else:
                    value = 0
                handle.writeframes(struct.pack("<h", value))
    return path


def test_normalize_asr_segments_sorts_and_cleans_text() -> None:
    segments, warnings = normalize_asr_segments(
        [
            AsrSegment(2, 1.0, 2.0, "  second   line "),
            AsrSegment(1, -1.0, 0.5, " first "),
        ]
    )

    assert [segment.id for segment in segments] == [1, 2]
    assert segments[0].start_sec == 0.0
    assert segments[0].text == "first"
    assert segments[1].text == "second line"
    assert warnings == []


def test_normalize_asr_segments_warns_on_empty_and_overlap() -> None:
    segments, warnings = normalize_asr_segments(
        [
            AsrSegment(1, 0, 2, "hello"),
            AsrSegment(2, 1, 3, "overlap"),
            AsrSegment(3, 3, 4, ""),
        ]
    )

    assert segments[1].start_sec == 2
    assert warnings == [
        "2: adjusted overlap from 1.000 to 2.000",
        "3: skipped empty text",
    ]


def test_asr_service_uses_provider_and_normalizes_result(tmp_path: Path) -> None:
    provider = ScriptedAsrProvider([AsrSegment(1, 0, 1, " hello ", speaker="SPEAKER_01")], language="en")
    service = AsrService(provider)

    result = service.transcribe(AsrRequest(audio_path=tmp_path / "raw.wav"))

    assert result.language == "en"
    assert result.segments[0].text == "hello"
    assert result.segments[0].speaker == "SPEAKER_01"
    assert provider.calls[0].audio_path == tmp_path / "raw.wav"


def test_asr_segments_to_subtitle_rows() -> None:
    rows = asr_segments_to_subtitle_rows([AsrSegment(1, 0, 1, "hello", speaker="SPEAKER_02")])

    assert rows[0].source == "hello"
    assert rows[0].target == ""
    assert rows[0].speaker == "SPEAKER_02"


def test_asr_stage_runner_returns_segments_and_rows(tmp_path: Path) -> None:
    runner = AsrStageRunner(ScriptedAsrProvider([AsrSegment(1, 0, 1, "hello")]))

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"audio_path": str(tmp_path / "raw.wav"), "source_language": "en"},
            stage=StageName.TRANSCRIBE,
            attempt=1,
        )
    )

    assert result.outputs["language"] == "en"
    assert result.outputs["segments"][0]["text"] == "hello"
    assert result.outputs["subtitle_rows"][0]["source"] == "hello"


def test_asr_stage_runner_skips_without_audio_path(tmp_path: Path) -> None:
    runner = AsrStageRunner(ScriptedAsrProvider())

    result = runner.run(StageContext("job", tmp_path, {}, StageName.TRANSCRIBE, 1))

    assert result.skipped
    assert result.warnings == ["No audio_path in task"]


def test_transcribe_stage_runner_extracts_audio_and_writes_rows(tmp_path: Path) -> None:
    media = FakeMediaProvider()
    runner = TranscribeStageRunner(ScriptedAsrProvider([AsrSegment(1, 0, 1, "hello")]), media)

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {"source_video": str(tmp_path / "source.mp4"), "output_dir": str(tmp_path / "output")},
            StageName.TRANSCRIBE,
            1,
        )
    )

    assert media.extract_calls[0].output_audio == tmp_path / "output" / "audio" / "raw_hq.wav"
    assert media.extract_calls[0].audio_codec == "pcm_s16le"
    assert media.extract_calls[1].output_audio == tmp_path / "output" / "audio" / "raw.mp3"
    assert media.extract_calls[1].audio_codec == "libmp3lame"
    assert media.extract_calls[1].audio_bitrate == "32k"
    assert result.outputs["raw_audio"].endswith("raw.mp3")
    assert result.outputs["high_quality_audio"].endswith("raw_hq.wav")
    assert Path(result.outputs["cleaned_chunks"]).exists()
    assert Path(result.outputs["split_by_nlp"]).read_text(encoding="utf-8").strip() == "hello"
    assert result.outputs["subtitle_rows"][0]["source"] == "hello"
    assert Path(result.outputs["subtitle_rows_json"]).exists()
    assert Path(result.outputs["subtitle_rows_json"]) == tmp_path / "output" / "internal" / "subtitle_rows.json"


def test_transcribe_stage_runner_reextracts_unusable_cached_audio(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "output"
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "raw_hq.wav").write_bytes(b"bad")
    (audio_dir / "raw.mp3").write_bytes(b"bad")
    media = FakeMediaProvider()
    runner = TranscribeStageRunner(ScriptedAsrProvider([AsrSegment(1, 0, 1, "hello")]), media)

    monkeypatch.setattr("eistara.core.asr.transcribe_runner.is_usable_media_file", lambda *_args, **_kwargs: False)

    runner.run(
        StageContext(
            "job",
            tmp_path,
            {"source_video": str(tmp_path / "source.mp4"), "output_dir": str(output_dir)},
            StageName.TRANSCRIBE,
            1,
        )
    )

    assert [call.output_audio for call in media.extract_calls] == [
        audio_dir / "raw_hq.wav",
        audio_dir / "raw.mp3",
    ]


def test_transcribe_stage_runner_uses_existing_audio(tmp_path: Path) -> None:
    media = FakeMediaProvider()
    runner = TranscribeStageRunner(ScriptedAsrProvider([AsrSegment(1, 0, 1, "hello")]), media)

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {"audio_path": str(tmp_path / "raw.wav"), "output_dir": str(tmp_path / "output")},
            StageName.TRANSCRIBE,
            1,
        )
    )

    assert media.extract_calls == []
    assert result.outputs["raw_audio"].endswith("raw.wav")
    assert Path(result.outputs["split_by_nlp"]).exists()


def test_transcribe_stage_runner_ignores_empty_existing_vocal(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    raw_audio = tmp_path / "raw.wav"
    raw_audio.write_bytes(b"raw")
    (audio_dir / "vocal.mp3").write_bytes(b"")
    provider = CapturingAsrProvider([AsrSegment(1, 0, 1, "hello")])
    runner = TranscribeStageRunner(provider, FakeMediaProvider())

    runner.run(
        StageContext(
            "job",
            tmp_path,
            {"audio_path": str(raw_audio), "output_dir": str(output_dir)},
            StageName.TRANSCRIBE,
            1,
        )
    )

    assert provider.settings_calls[0].provider_config["vocal_audio_path"] == str(raw_audio)


def test_transcribe_stage_runner_runs_v1_demucs_when_configured(tmp_path: Path) -> None:
    media = FakeMediaProvider()
    demucs = FakeVocalSeparationProvider()
    runner = TranscribeStageRunner(
        ScriptedAsrProvider([AsrSegment(1, 0, 1, "hello")]),
        media,
        AsrSettings(provider_config={"demucs": True, "demucs_segment_minutes": 12}),
        vocal_separation_provider=demucs,
    )

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {"source_video": str(tmp_path / "source.mp4"), "output_dir": str(tmp_path / "output")},
            StageName.TRANSCRIBE,
            1,
        )
    )

    assert result.outputs["vocal_audio"].endswith("vocal.mp3")
    assert result.outputs["background_audio"].endswith("background.mp3")
    assert demucs.separate_calls[0][2] == 12
    assert demucs.normalize_calls[0][0] == tmp_path / "output" / "audio" / "vocal.mp3"


def test_detect_audio_pauses_finds_low_energy_gap(tmp_path: Path) -> None:
    audio = write_tone_silence_wav(
        tmp_path / "pause.wav",
        [("tone", 0.5), ("silence", 0.7), ("tone", 0.5)],
    )

    pauses, report = detect_audio_pauses(audio, provider_config={"source_subtitle_audio_pause_min_sec": 0.4})

    assert report["available"] is True
    assert pauses is not None
    assert len(pauses) == 1
    assert pauses[0].start_sec <= 0.55
    assert pauses[0].end_sec >= 1.15


def test_source_subtitle_cleanup_does_not_split_on_audio_pause_without_subtitle_gap() -> None:
    rows = [
        SubtitleRow(0.0, 0.2, "This", ""),
        SubtitleRow(0.2, 0.4, "should", ""),
        SubtitleRow(0.4, 0.6, "stay", ""),
        SubtitleRow(0.6, 0.8, "together", ""),
    ]

    result = normalize_source_subtitle_rows(
        rows,
        audio_pauses=[AudioPause(0.25, 0.7)],
    )

    assert [row.source for row in result.rows] == ["This should stay together"]
    assert result.report["boundary_reasons"] == {"end_of_input": 1}


def test_transcribe_stage_runner_fails_when_extract_fails(tmp_path: Path) -> None:
    media = FakeMediaProvider(MediaCommandResult(("extract",), 1, stderr="no audio"))
    runner = TranscribeStageRunner(ScriptedAsrProvider([AsrSegment(1, 0, 1, "hello")]), media)

    try:
        runner.run(StageContext("job", tmp_path, {"source_video": "source.mp4"}, StageName.TRANSCRIBE, 1))
    except RuntimeError as exc:
        assert "no audio" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
