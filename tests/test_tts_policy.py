from __future__ import annotations

import json
import io
import math
from pathlib import Path
import struct
import wave

import pandas as pd

from eistara.core.jobs import StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.media import AudioStreamInfo, MediaInfo
from eistara.core.pipeline import StageContext
from eistara.core.scheduler import SchedulerService
from eistara.core.tts.audio import wav_duration_sec
from eistara.core.tts.speed import v1_slow_fast_speed_factor
from eistara.core.tts import (
    ScriptedTtsProvider,
    TtsCachePolicy,
    TtsRequest,
    TtsPrepareStageRunner,
    TtsProviderError,
    TtsService,
    TtsServiceError,
    TtsSettings,
    TtsStageRunner,
    cache_meta_path,
    clean_text_for_tts,
)


class FakeMediaProbe:
    def __init__(self, duration: float = 1.25):
        self.duration = duration

    def probe(self, path: str) -> MediaInfo:
        return MediaInfo(path=Path(path), duration_sec=self.duration, audio=AudioStreamInfo(duration_sec=self.duration))


class ProbeableScriptedTtsProvider(ScriptedTtsProvider):
    def __init__(self, failures: list[Exception] | None = None):
        super().__init__(failures=failures)
        self.ready_checks = 0

    def check_ready(self, settings: TtsSettings) -> None:
        self.ready_checks += 1


class CapturingPrepareSettingsTtsProvider(ScriptedTtsProvider):
    def __init__(self):
        super().__init__()
        self.prepare_calls = []

    def prepare_settings(self, settings: TtsSettings, *, output_dir, reference_audio_dir=None) -> TtsSettings:
        self.prepare_calls.append((Path(output_dir), Path(reference_audio_dir) if reference_audio_dir else None))
        return settings


def test_clean_text_for_tts_folds_latin_diacritics() -> None:
    assert clean_text_for_tts("Björn™ & São") == "Bjorn  Sao"


def test_clean_text_for_tts_removes_decomposed_latin_diacritics() -> None:
    assert clean_text_for_tts("c\u0327a Fran\u0063\u0327ois") == "ca Francois"


def test_clean_text_for_tts_matches_v1_special_latin_fold_table() -> None:
    assert clean_text_for_tts("\u0110\u0111") == "Dd"
    assert clean_text_for_tts("\u00d0\u00f0") == "\u00d0\u00f0"


def test_tts_cache_signature_changes_with_provider_config(tmp_path: Path) -> None:
    request = TtsRequest(text="hello", output_path=tmp_path / "a.wav")
    first = TtsCachePolicy(TtsSettings(method="openai_tts", provider_config={"voice": "a"})).build_metadata(request, "hello")
    second = TtsCachePolicy(TtsSettings(method="openai_tts", provider_config={"voice": "b"})).build_metadata(request, "hello")

    assert first["signature"] != second["signature"]


def test_tts_cache_metadata_uses_v1_cache_version_and_indextts_prompt_fingerprint(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.wav"
    _write_test_wav(prompt, duration_sec=0.2)
    request = TtsRequest(text="hello", output_path=tmp_path / "a.wav")

    metadata = TtsCachePolicy(
        TtsSettings(
            provider_config={
                "prompt_audio_mode": "auto_ref",
                "prompt_audio": str(prompt),
                "output_dir": str(tmp_path / "output-a"),
                "reference_audio_dir": str(tmp_path / "refs-a"),
                "top_p": 0.8,
            },
            audio_config={"postprocess_audio": True, "ffmpeg_path": "ffmpeg-a"},
        )
    ).build_metadata(request, "hello")

    assert metadata["payload"]["version"] == 2
    assert metadata["payload"]["tts_method"] == "indextts"
    assert metadata["payload"]["indextts"]["effective_prompt_audio"] == str(prompt)
    assert metadata["payload"]["indextts"]["effective_prompt_audio_file"]["exists"] is True
    assert "output_dir" not in metadata["payload"]["indextts"]["config"]
    assert "reference_audio_dir" not in metadata["payload"]["indextts"]["config"]
    assert "ffmpeg_path" not in metadata["payload"]["tts_audio"]


def test_tts_cache_signature_ignores_runtime_paths_but_changes_with_prompt_audio(tmp_path: Path) -> None:
    first_prompt = tmp_path / "first.wav"
    second_prompt = tmp_path / "second.wav"
    _write_test_wav(first_prompt, duration_sec=0.2)
    _write_test_wav(second_prompt, duration_sec=0.3)
    request = TtsRequest(text="hello", output_path=tmp_path / "a.wav")

    first = TtsCachePolicy(
        TtsSettings(
            provider_config={
                "prompt_audio_mode": "auto_ref",
                "prompt_audio": str(first_prompt),
                "output_dir": str(tmp_path / "output-a"),
                "reference_audio_dir": str(tmp_path / "refs-a"),
            },
            audio_config={"postprocess_audio": True, "ffmpeg_path": "ffmpeg-a"},
        )
    ).build_metadata(request, "hello")
    same_prompt_other_paths = TtsCachePolicy(
        TtsSettings(
            provider_config={
                "prompt_audio_mode": "auto_ref",
                "prompt_audio": str(first_prompt),
                "output_dir": str(tmp_path / "output-b"),
                "reference_audio_dir": str(tmp_path / "refs-b"),
            },
            audio_config={"postprocess_audio": True, "ffmpeg_path": "ffmpeg-b"},
        )
    ).build_metadata(request, "hello")
    other_prompt = TtsCachePolicy(
        TtsSettings(provider_config={"prompt_audio_mode": "auto_ref", "prompt_audio": str(second_prompt)})
    ).build_metadata(request, "hello")

    assert first["signature"] == same_prompt_other_paths["signature"]
    assert first["signature"] != other_prompt["signature"]


def test_tts_service_writes_cache_and_skips_second_call(tmp_path: Path) -> None:
    provider = ScriptedTtsProvider()
    service = TtsService(provider)
    request = TtsRequest(text="hello", output_path=tmp_path / "hello.wav")

    first = service.synthesize(request)
    second = service.synthesize(request)

    assert first.cached is False
    assert second.cached is True
    assert len(provider.calls) == 1
    assert cache_meta_path(request.output_path).exists()


def test_tts_service_adopts_legacy_audio_without_cache(tmp_path: Path) -> None:
    output = tmp_path / "legacy.wav"
    _write_test_wav(output, duration_sec=0.2)
    original = output.read_bytes()
    provider = ScriptedTtsProvider()
    service = TtsService(provider)

    result = service.synthesize(TtsRequest(text="hello", output_path=output))

    assert result.cached is True
    assert output.read_bytes() == original
    assert len(provider.calls) == 0
    metadata = json.loads(cache_meta_path(output).read_text(encoding="utf-8"))
    assert metadata["legacy_adopted"] is True


def test_tts_service_writes_v1_silence_for_empty_or_single_char_text(tmp_path: Path) -> None:
    provider = ScriptedTtsProvider()
    service = TtsService(provider)
    output = tmp_path / "silent.wav"

    result = service.synthesize(TtsRequest(text="!", output_path=output))

    assert len(provider.calls) == 0
    assert result.warnings == ["silent placeholder"]
    duration = wav_duration_sec(output)
    assert duration is not None
    assert 0.09 <= duration <= 0.11
    assert cache_meta_path(output).exists()


def test_tts_service_retries_zero_duration_then_writes_v1_silence(tmp_path: Path) -> None:
    provider = ScriptedTtsProvider(payload=b"")
    service = TtsService(provider, TtsSettings(max_retries=2), sleep=lambda _seconds: None)
    output = tmp_path / "zero.wav"

    result = service.synthesize(TtsRequest(text="hello", output_path=output))

    assert len(provider.calls) == 2
    assert result.warnings == ["zero-duration placeholder"]
    duration = wav_duration_sec(output)
    assert duration is not None
    assert 0.09 <= duration <= 0.11
    assert cache_meta_path(output).exists()


def test_tts_service_retries_service_errors(tmp_path: Path) -> None:
    provider = ScriptedTtsProvider(failures=[TtsServiceError("down")])
    sleeps: list[float] = []
    service = TtsService(provider, TtsSettings(max_retries=2, service_backoff_base_sec=0.01), sleep=sleeps.append)

    service.synthesize(TtsRequest(text="hello", output_path=tmp_path / "hello.wav"))

    assert len(provider.calls) == 2
    assert sleeps


def test_tts_service_reprobes_provider_after_service_error(tmp_path: Path) -> None:
    provider = ProbeableScriptedTtsProvider(failures=[TtsServiceError("down")])
    service = TtsService(
        provider,
        TtsSettings(max_retries=2, service_backoff_base_sec=0.01),
        sleep=lambda _seconds: None,
    )

    service.synthesize(TtsRequest(text="hello", output_path=tmp_path / "hello.wav"))

    assert provider.ready_checks == 1
    assert [call.text for call in provider.calls] == ["hello", "hello"]


def test_tts_service_corrects_text_only_on_final_content_retry(tmp_path: Path) -> None:
    provider = ScriptedTtsProvider(failures=[TtsProviderError("bad text")])
    service = TtsService(
        provider,
        TtsSettings(max_retries=2),
        sleep=lambda _seconds: None,
        text_corrector=lambda text: f"{text} fixed",
    )

    service.synthesize(TtsRequest(text="hello", output_path=tmp_path / "hello.wav"))

    assert [call.text for call in provider.calls] == ["hello", "hello fixed"]


def test_tts_service_applies_v1_generated_audio_postprocess(tmp_path: Path) -> None:
    from pydub import AudioSegment
    from pydub.generators import Sine

    audio = (
        AudioSegment.silent(duration=400)
        + Sine(440).to_audio_segment(duration=500).apply_gain(-1)
        + AudioSegment.silent(duration=400)
    )
    provider = ScriptedTtsProvider(payload=_audio_bytes(audio))
    service = TtsService(
        provider,
        TtsSettings(
            audio_config={
                "postprocess_audio": True,
                "trim_silence": True,
                "trim_silence_padding_ms": 50,
                "trim_min_silence_len_ms": 120,
                "trim_silence_threshold_offset_db": 22,
                "trim_silence_threshold_min_dbfs": -45,
                "peak_normalize_dbfs": -6.0,
            }
        ),
    )
    output = tmp_path / "processed.wav"

    result = service.synthesize(TtsRequest(text="hello", output_path=output))

    processed = AudioSegment.from_file(output)
    assert result.warnings == []
    assert len(processed) < 750
    assert processed.max_dBFS <= -5.9


def test_v1_slow_fast_speed_factor_slows_fast_readability_segments() -> None:
    config = {
        "slow_fast_segments": True,
        "slow_fast_trigger_units_per_sec": 5.75,
        "slow_fast_target_units_per_sec": 5.4,
        "slow_fast_max_duration_factor": 1.1,
        "slow_fast_min_duration_sec": 1.2,
    }

    speed = v1_slow_fast_speed_factor("one two three four five six seven eight nine ten", 2.0, config)

    assert speed == 0.909
    assert v1_slow_fast_speed_factor("one two three four five six seven eight nine ten", 2.0, {**config, "slow_fast_segments": False}) == 1.0


def test_tts_stage_runner_updates_scheduler_outputs(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job_0001_tts"
    job_dir.mkdir(parents=True)
    (job_dir / TASK_FILE).write_text(
        json.dumps(
            {
                "id": job_dir.name,
                "tts_segments": [
                    {"id": "1", "text": "你好", "output_path": "output/audio/1.wav"},
                    {"id": "2", "text": "世界", "output_path": "output/audio/2.wav"},
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": ["download", "transcribe", "translate", "tts_prepare"],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    provider = ScriptedTtsProvider()
    service = SchedulerService(jobs_dir)
    service.register(TtsStageRunner(provider))

    assert service.run_one_ready_stage() is True
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    assert state["completed_stages"][-1] == "tts"
    assert state["artifacts"]["tts_count"] == 2
    assert sorted(state["artifacts"]["tts_outputs"]) == ["1", "2"]
    assert Path(state["artifacts"]["tts_audio_quality_report"]).exists()
    assert "tts_segments" not in state["artifacts"]
    assert len(provider.calls) == 2


def test_tts_stage_runner_writes_audio_quality_report(tmp_path: Path) -> None:
    runner = TtsStageRunner(ScriptedTtsProvider())

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {"tts_segments": [{"id": "1", "text": "hello", "output_path": "output/audio/tmp/1_0_temp.wav"}]},
            StageName.TTS,
            1,
        )
    )

    report = Path(result.outputs["tts_audio_quality_report"])
    assert report == tmp_path / "output" / "log" / "tts_audio_quality.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["summary"]["segment_count"] == 1
    assert data["segments"][0]["segment_id"] == "1"
    assert data["segments"][0]["audio_path"].endswith("1_0_temp.wav")


def test_tts_stage_runner_passes_job_scoped_output_dir_to_provider(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobA"
    provider = CapturingPrepareSettingsTtsProvider()
    runner = TtsStageRunner(provider)

    runner.run(
        StageContext(
            "jobA",
            job_dir,
            {"tts_segments": [{"id": "1", "text": "hello", "output_path": "output/audio/tmp/1_0_temp.wav"}]},
            StageName.TTS,
            1,
        )
    )

    assert provider.prepare_calls[0][0] == job_dir / "output"
    assert provider.calls[0].output_path == job_dir / "output" / "audio" / "tmp" / "1_0_temp.wav"


def test_tts_stage_runner_writes_real_duration_to_v1_task_sheet(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "text": "hello"}]).to_excel(tts_tasks, index=False)
    runner = TtsStageRunner(ScriptedTtsProvider(), media_probe=FakeMediaProbe(1.75))

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {
                "tts_tasks": str(tts_tasks),
                "tts_segments": [{"id": "1", "text": "hello", "output_path": str(tmp_path / "output" / "audio" / "tmp" / "1_0_temp.wav")}],
            },
            StageName.TTS,
            1,
        )
    )

    assert result.outputs["tts_durations"] == {"1": 1.75}
    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert row["real_dur"] == 1.75


def test_tts_stage_runner_sums_line_durations_to_v1_task_sheet(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "text": "hello"}]).to_excel(tts_tasks, index=False)
    runner = TtsStageRunner(ScriptedTtsProvider(), media_probe=FakeMediaProbe(1.75))

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {
                "tts_tasks": str(tts_tasks),
                "tts_segments": [
                    {"id": "1_0", "text": "hello", "output_path": str(tmp_path / "output" / "audio" / "tmp" / "1_0_temp.wav")},
                    {"id": "1_1", "text": "world", "output_path": str(tmp_path / "output" / "audio" / "tmp" / "1_1_temp.wav")},
                ],
            },
            StageName.TTS,
            1,
        )
    )

    assert result.outputs["tts_durations"] == {"1_0": 1.75, "1_1": 1.75}
    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert row["real_dur"] == 3.5


def test_tts_prepare_stage_runner_writes_v1_task_sheet(tmp_path: Path) -> None:
    runner = TtsPrepareStageRunner()

    result = runner.run(
        StageContext(
            "job",
            tmp_path,
            {
                "output_dir": str(tmp_path / "output"),
                "tts_segments": [
                    {
                        "id": 1,
                        "start": 0.0,
                        "end": 1.5,
                        "source": "Hello world",
                        "text": "浣犲ソ",
                    }
                ],
            },
            StageName.TTS_PREPARE,
            1,
        )
    )

    task_path = Path(result.outputs["tts_tasks"])
    assert task_path.exists()
    row = pd.read_excel(task_path).to_dict(orient="records")[0]
    assert row["number"] == 1
    assert row["start_time"] == "00:00:00.000"
    assert row["end_time"] == "00:00:01.500"
    assert row["text"] == "浣犲ソ"
    assert row["origin"] == "Hello world"
    assert result.outputs["tts_segments_count"] == 1
    assert result.outputs["tts_segments"][0]["output_path"].endswith("output\\audio\\tmp\\1_0_temp.wav") or result.outputs[
        "tts_segments"
    ][0]["output_path"].endswith("output/audio/tmp/1_0_temp.wav")


def test_tts_prepare_stage_runner_expands_v1_lines_to_tts_segments(tmp_path: Path) -> None:
    result = TtsPrepareStageRunner().run(
        StageContext(
            "job",
            tmp_path,
            {
                "output_dir": str(tmp_path / "output"),
                "tts_segments": [
                    {
                        "id": 1,
                        "start": 0.0,
                        "end": 2.0,
                        "source": "Hello world",
                        "text": "Hi there",
                        "lines": ["Hi", "there"],
                        "src_lines": ["Hello", "world"],
                    }
                ],
            },
            StageName.TTS_PREPARE,
            1,
        )
    )

    segments = result.outputs["tts_segments"]
    assert [segment["id"] for segment in segments] == ["1_0", "1_1"]
    assert [segment["text"] for segment in segments] == ["Hi", "there"]
    assert segments[0]["output_path"].endswith("output\\audio\\tmp\\1_0_temp.wav") or segments[0]["output_path"].endswith(
        "output/audio/tmp/1_0_temp.wav"
    )
    assert segments[1]["output_path"].endswith("output\\audio\\tmp\\1_1_temp.wav") or segments[1]["output_path"].endswith(
        "output/audio/tmp/1_1_temp.wav"
    )


def test_tts_prepare_stage_runner_applies_v1_micro_line_merge(tmp_path: Path) -> None:
    result = TtsPrepareStageRunner(audio_config={"merge_micro_lines": True, "merge_micro_line_chars": 6}).run(
        StageContext(
            "job",
            tmp_path,
            {
                "output_dir": str(tmp_path / "output"),
                "tts_segments": [
                    {
                        "id": 1,
                        "start": 0.0,
                        "end": 2.0,
                        "source": "Source",
                        "text": "A longer sentence",
                        "lines": ["A", "longer sentence"],
                        "src_lines": ["Src A", "Src sentence"],
                    }
                ],
            },
            StageName.TTS_PREPARE,
            1,
        )
    )

    segments = result.outputs["tts_segments"]
    assert [segment["id"] for segment in segments] == ["1_0"]
    assert segments[0]["text"] == "A longer sentence"
    task_row = pd.read_excel(result.outputs["tts_tasks"]).to_dict(orient="records")[0]
    assert task_row["lines"] == "['A longer sentence']"
    assert task_row["src_lines"] == "['Src A Src sentence']"
    report = Path(result.outputs["micro_tts_line_merge_report"])
    assert report.exists()
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["changed_rows"] == 1
    assert data["changes"][0]["groups"] == [[0, 1]]


def test_tts_prepare_stage_runner_reads_v1_audio_subtitles(tmp_path: Path) -> None:
    audio_dir = tmp_path / "output" / "audio"
    audio_dir.mkdir(parents=True)
    (audio_dir / "trans_subs_for_audio.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,500\nHello (aside)-world\n\n",
        encoding="utf-8",
    )
    (audio_dir / "src_subs_for_audio.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,500\nSource line\n\n",
        encoding="utf-8",
    )

    result = TtsPrepareStageRunner().run(
        StageContext(
            "job",
            tmp_path,
            {"output_dir": str(tmp_path / "output")},
            StageName.TTS_PREPARE,
            1,
        )
    )

    row = pd.read_excel(result.outputs["tts_tasks"]).to_dict(orient="records")[0]
    assert row["number"] == 1
    assert row["start_time"] == "00:00:00.000"
    assert row["end_time"] == "00:00:01.500"
    assert row["text"] == "Hello world"
    assert row["origin"] == "Source line"
    assert result.outputs["tts_segments"][0]["text"] == "Hello world"


def test_tts_prepare_stage_runner_extracts_reference_audio_from_vocal(tmp_path: Path) -> None:
    vocal = tmp_path / "vocal.wav"
    _write_test_wav(vocal, duration_sec=2.0)

    result = TtsPrepareStageRunner().run(
        StageContext(
            "job",
            tmp_path,
            {
                "output_dir": str(tmp_path / "output"),
                "vocal_audio": str(vocal),
                "tts_segments": [{"id": 1, "start": 0.0, "end": 1.0, "source": "src", "text": "hello"}],
            },
            StageName.TTS_PREPARE,
            1,
        )
    )

    assert result.warnings == []
    assert (Path(result.outputs["reference_audio_dir"]) / "1.wav").exists()


def test_tts_prepare_reference_audio_stays_under_job_relative_output_dir(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobA"
    vocal = job_dir / "input" / "vocal.wav"
    _write_test_wav(vocal, duration_sec=2.0)

    result = TtsPrepareStageRunner().run(
        StageContext(
            "jobA",
            job_dir,
            {
                "output_dir": "custom_output",
                "vocal_audio": str(vocal),
                "tts_segments": [{"id": 1, "start": 0.0, "end": 1.0, "source": "src", "text": "hello"}],
            },
            StageName.TTS_PREPARE,
            1,
        )
    )

    reference_dir = Path(result.outputs["reference_audio_dir"])
    assert reference_dir == job_dir / "custom_output" / "audio" / "refers"
    assert (reference_dir / "1.wav").exists()
    assert not (Path.cwd() / "custom_output" / "audio" / "refers" / "1.wav").exists()


def test_tts_prepare_stage_runner_does_not_use_raw_hq_for_reference_when_vocal_missing(tmp_path: Path) -> None:
    raw_hq = tmp_path / "output" / "audio" / "raw_hq.wav"
    _write_test_wav(raw_hq, duration_sec=2.0)

    result = TtsPrepareStageRunner().run(
        StageContext(
            "job",
            tmp_path,
            {
                "output_dir": str(tmp_path / "output"),
                "high_quality_audio": str(raw_hq),
                "tts_segments": [{"id": 1, "start": 0.0, "end": 1.0, "source": "src", "text": "hello"}],
            },
            StageName.TTS_PREPARE,
            1,
        )
    )

    reference = Path(result.outputs["reference_audio_dir"]) / "1.wav"
    assert result.warnings == [f"reference audio extraction skipped: vocal_audio does not exist: {tmp_path / 'output' / 'audio' / 'vocal.mp3'}"]
    assert not reference.exists()


def _write_test_wav(path: Path, duration_sec: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration_sec * sample_rate)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(frames):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            handle.writeframes(struct.pack("<h", value))


def _audio_bytes(audio) -> bytes:
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()
