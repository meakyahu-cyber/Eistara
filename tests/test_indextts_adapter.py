from __future__ import annotations

import io
import json
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

import pytest

from eistara.adapters.tts import IndexTtsProvider, build_indextts_payload, indextts_root_url, prepare_indextts_prompt_audio
from eistara.core.jobs import StageName
from eistara.core.pipeline import StageContext
from eistara.core.tts import TtsProviderError, TtsRequest, TtsService, TtsServiceError, TtsSettings
from eistara.core.tts.runner import TtsStageRunner


@dataclass(slots=True)
class FakeResponse:
    status_code: int = 200
    content: bytes = b"audio"
    text: str = ""

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeTransport:
    def __init__(self, post_response: FakeResponse | None = None, post_error: Exception | None = None):
        self.post_response = post_response or FakeResponse()
        self.post_error = post_error
        self.get_calls: list[tuple[str, float]] = []
        self.post_calls: list[dict] = []

    def get(self, url: str, *, timeout: float) -> FakeResponse:
        self.get_calls.append((url, timeout))
        return FakeResponse()

    def post(self, url: str, *, json: dict, timeout: float) -> FakeResponse:
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        if self.post_error:
            raise self.post_error
        return self.post_response


class DurationAwareTransport(FakeTransport):
    def __init__(self, natural_durations: dict[str, float]):
        super().__init__()
        self.natural_durations = dict(natural_durations)

    def post(self, url: str, *, json: dict, timeout: float) -> FakeResponse:
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        duration = float(json.get("target_duration_sec") or self.natural_durations.get(str(json.get("text")), 0.5))
        return FakeResponse(content=_wav_bytes(duration_ms=round(duration * 1000)))


def test_indextts_root_url() -> None:
    assert indextts_root_url("http://127.0.0.1:8010/tts") == "http://127.0.0.1:8010"


def test_build_payload_omits_empty_prompt_audio() -> None:
    payload = build_indextts_payload("hello", {"prompt_audio": "", "top_p": 0.7})

    assert payload["text"] == "hello"
    assert payload["top_p"] == 0.7
    assert "prompt_audio" not in payload


def test_build_payload_includes_duration_control_when_enabled() -> None:
    payload = build_indextts_payload(
        "hello",
        {},
        duration_control={
            "duration_control_enabled": True,
            "target_duration_sec": 0.72,
        },
    )

    assert payload["duration_control_enabled"] is True
    assert payload["target_duration_sec"] == 0.72
    assert "duration_scale" not in payload


def test_build_payload_does_not_apply_provider_duration_control_globally() -> None:
    payload = build_indextts_payload(
        "hello",
        {
            "duration_control": {
                "enabled": True,
                "target_duration_sec": 0.72,
            }
        },
    )

    assert "duration_control_enabled" not in payload
    assert "target_duration_sec" not in payload


def test_check_ready_uses_service_root() -> None:
    transport = FakeTransport()
    provider = IndexTtsProvider(transport)

    provider.check_ready(TtsSettings(provider_config={"api_url": "http://localhost:8010/tts"}), timeout=2)

    assert transport.get_calls == [("http://localhost:8010", 2)]


def test_indextts_success_writes_audio(tmp_path: Path) -> None:
    payload = _wav_bytes()
    transport = FakeTransport(post_response=FakeResponse(content=payload))
    provider = IndexTtsProvider(transport)
    output = tmp_path / "out.wav"

    provider.synthesize(TtsRequest(text="hello", output_path=output), TtsSettings(provider_config={"api_url": "http://x/tts"}))

    assert output.read_bytes() == payload
    assert transport.post_calls[0]["url"] == "http://x/tts"
    assert transport.post_calls[0]["json"]["text"] == "hello"


def test_indextts_duration_control_request_metadata_enables_per_request_control(tmp_path: Path) -> None:
    payload = _wav_bytes()
    transport = FakeTransport(post_response=FakeResponse(content=payload))
    provider = IndexTtsProvider(transport)
    output = tmp_path / "out.wav"

    provider.synthesize(
        TtsRequest(
            text="hello",
            output_path=output,
            metadata={"indextts_duration_control": {"enabled": True, "duration_scale": 0.82}},
        ),
        TtsSettings(
            provider_config={
                "api_url": "http://x/tts",
                "duration_control": {"enabled": True, "target_duration_sec": 1.4},
            }
        ),
    )

    posted = transport.post_calls[0]["json"]
    assert posted["duration_control_enabled"] is True
    assert posted["duration_scale"] == 0.82
    assert "target_duration_sec" not in posted


def test_indextts_5xx_is_service_error(tmp_path: Path) -> None:
    provider = IndexTtsProvider(FakeTransport(post_response=FakeResponse(status_code=500, text="boom")))

    try:
        provider.synthesize(TtsRequest(text="hello", output_path=tmp_path / "out.wav"), TtsSettings())
    except TtsServiceError as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("expected TtsServiceError")


def test_indextts_4xx_is_provider_error(tmp_path: Path) -> None:
    provider = IndexTtsProvider(FakeTransport(post_response=FakeResponse(status_code=400, text="bad text")))

    try:
        provider.synthesize(TtsRequest(text="hello", output_path=tmp_path / "out.wav"), TtsSettings())
    except TtsProviderError as exc:
        assert "400" in str(exc)
    else:
        raise AssertionError("expected TtsProviderError")


def test_indextts_adapter_works_with_tts_service(tmp_path: Path) -> None:
    payload = _wav_bytes()
    transport = FakeTransport(post_response=FakeResponse(content=payload))
    service = TtsService(IndexTtsProvider(transport), TtsSettings(provider_config={"api_url": "http://x/tts"}))

    result = service.synthesize(TtsRequest(text="Björn", output_path=tmp_path / "out.wav"))

    assert result.output_path.read_bytes() == payload
    assert transport.post_calls[0]["json"]["text"] == "Bjorn"


def test_indextts_auto_ref_prompt_audio_is_generated_from_references(tmp_path: Path) -> None:
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    reference_dir = output_dir / "audio" / "refers"
    reference_dir.mkdir(parents=True)
    Sine(440).to_audio_segment(duration=7000).apply_gain(-12).export(reference_dir / "1.wav", format="wav")
    config = {
        "prompt_audio_mode": "auto_ref",
        "output_dir": str(output_dir),
        "reference_audio_dir": str(reference_dir),
        "auto_prompt_target_sec": 6,
        "auto_prompt_min_prompt_sec": 5,
    }

    prompt_audio = prepare_indextts_prompt_audio(config)

    assert prompt_audio.endswith("indextts_prompt.wav")
    assert Path(prompt_audio).exists()
    assert (output_dir / "log" / "indextts_prompt_audio.json").exists()


def test_indextts_auto_ref_extracts_references_from_vocal_before_prompt_selection(tmp_path: Path) -> None:
    import pandas as pd
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    audio_dir = output_dir / "audio"
    vocal = audio_dir / "vocal.wav"
    tts_tasks = audio_dir / "tts_tasks.xlsx"
    audio_dir.mkdir(parents=True)
    Sine(440).to_audio_segment(duration=7000).apply_gain(-12).export(vocal, format="wav")
    pd.DataFrame(
        [
            {
                "number": 1,
                "start_time": "00:00:00.000",
                "end_time": "00:00:07.000",
            }
        ]
    ).to_excel(tts_tasks, index=False)

    prompt_audio = prepare_indextts_prompt_audio(
        {
            "prompt_audio_mode": "auto_ref",
            "output_dir": str(output_dir),
            "vocal_audio": str(vocal),
            "tts_tasks": str(tts_tasks),
            "auto_prompt_target_sec": 6,
            "auto_prompt_min_prompt_sec": 5,
        }
    )

    assert (audio_dir / "refers" / "1.wav").exists()
    assert prompt_audio.endswith("indextts_prompt.wav")
    assert Path(prompt_audio).exists()


def test_indextts_prepare_settings_adds_effective_prompt_audio(tmp_path: Path) -> None:
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    reference_dir = output_dir / "audio" / "refers"
    reference_dir.mkdir(parents=True)
    Sine(440).to_audio_segment(duration=7000).apply_gain(-12).export(reference_dir / "1.wav", format="wav")
    provider = IndexTtsProvider(FakeTransport())

    settings = provider.prepare_settings(
        TtsSettings(
            provider_config={
                "prompt_audio_mode": "auto_ref",
                "auto_prompt_target_sec": 6,
                "auto_prompt_min_prompt_sec": 5,
            }
        ),
        output_dir=output_dir,
        reference_audio_dir=reference_dir,
    )

    assert settings.provider_config["prompt_audio"].endswith("indextts_prompt.wav")
    assert build_indextts_payload("hello", settings.provider_config)["prompt_audio"] == settings.provider_config["prompt_audio"]


def test_tts_stage_runner_sends_job_local_indextts_prompt_audio(tmp_path: Path) -> None:
    from pydub.generators import Sine

    job_dir = tmp_path / "jobA"
    output_dir = job_dir / "output"
    reference_dir = output_dir / "audio" / "refers"
    reference_dir.mkdir(parents=True)
    Sine(440).to_audio_segment(duration=7000).apply_gain(-12).export(reference_dir / "1.wav", format="wav")
    transport = FakeTransport(post_response=FakeResponse(content=_wav_bytes(duration_ms=250)))
    runner = TtsStageRunner(
        provider=IndexTtsProvider(transport),
        settings=TtsSettings(
            max_retries=1,
            provider_config={
                "api_url": "http://x/tts",
                "prompt_audio_mode": "auto_ref",
                "auto_prompt_target_sec": 6,
                "auto_prompt_min_prompt_sec": 5,
            },
            audio_config={"postprocess_audio": False},
        ),
    )

    result = runner.run(
        StageContext(
            job_id="jobA",
            job_dir=job_dir,
            task={
                "output_dir": str(output_dir),
                "reference_audio_dir": str(reference_dir),
                "tts_segments": [
                    {
                        "id": "1_0",
                        "text": "hello",
                        "output_path": str(output_dir / "audio" / "tmp" / "1_0_temp.wav"),
                    }
                ],
            },
            stage=StageName.TTS,
            attempt=1,
        )
    )

    prompt_audio = transport.post_calls[0]["json"]["prompt_audio"]
    assert prompt_audio == str(reference_dir / "indextts_prompt.wav")
    assert Path(prompt_audio).exists()
    assert str(job_dir) in prompt_audio
    assert result.outputs["tts_count"] == 1


def test_indextts_stage_runner_retries_only_hard_source_window_with_duration_control(tmp_path: Path) -> None:
    transport = DurationAwareTransport({"front": 1.0, "hard": 1.6})
    runner = TtsStageRunner(
        provider=IndexTtsProvider(transport),
        settings=TtsSettings(
            max_retries=1,
            provider_config={
                "api_url": "http://x/tts",
                "duration_control": {
                    "enabled": False,
                    "adaptive_source_window_retry": {
                        "enabled": True,
                        "screen_borrow_max_sec": 0.3,
                        "screen_max_audio_speed": 1.07,
                        "target_borrow_max_sec": 0.5,
                        "borrow_min_seam_sec": 0.12,
                        "max_window_gap_sec": 0.0,
                    },
                },
            },
            audio_config={"postprocess_audio": False},
        ),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "tts_segments": [
                    {"id": "1_0", "number": 1, "start": 0.0, "end": 1.0, "text": "front", "output_path": "output/audio/tmp/1_0_temp.wav"},
                    {"id": "2_0", "number": 2, "start": 1.8, "end": 2.8, "text": "hard", "output_path": "output/audio/tmp/2_0_temp.wav"},
                ]
            },
            stage=StageName.TTS,
            attempt=1,
        )
    )

    assert len(transport.post_calls) == 3
    assert "duration_control_enabled" not in transport.post_calls[0]["json"]
    assert "duration_control_enabled" not in transport.post_calls[1]["json"]
    retry_payload = transport.post_calls[2]["json"]
    assert retry_payload["text"] == "hard"
    assert retry_payload["duration_control_enabled"] is True
    assert retry_payload["target_duration_sec"] == 1.5
    assert result.outputs["tts_provider_retry_count"] == 1
    assert result.outputs["tts_durations"]["2_0"] == pytest.approx(1.5)

    report = json.loads(Path(result.outputs["tts_audio_quality_report"]).read_text(encoding="utf-8"))
    retry_info = {row["segment_id"]: row["provider_retry"] for row in report["segments"]}
    assert retry_info["1_0"] is None
    assert retry_info["2_0"]["retry_kind"] == "indextts_duration_control"
    assert retry_info["2_0"]["screen_borrow_sec"] == pytest.approx(0.3)
    assert retry_info["2_0"]["target_borrow_sec"] == pytest.approx(0.5)


def test_indextts_stage_runner_does_not_retry_when_small_clip_speed_is_enough(tmp_path: Path) -> None:
    transport = DurationAwareTransport({"small": 1.05})
    runner = TtsStageRunner(
        provider=IndexTtsProvider(transport),
        settings=TtsSettings(
            max_retries=1,
            provider_config={
                "api_url": "http://x/tts",
                "duration_control": {
                    "enabled": False,
                    "adaptive_source_window_retry": {
                        "enabled": True,
                        "screen_max_audio_speed": 1.07,
                        "max_window_gap_sec": 0.0,
                    },
                },
            },
            audio_config={"postprocess_audio": False},
        ),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "tts_segments": [
                    {"id": "1_0", "number": 1, "start": 0.0, "end": 1.0, "text": "small", "output_path": "output/audio/tmp/1_0_temp.wav"},
                ]
            },
            stage=StageName.TTS,
            attempt=1,
        )
    )

    assert len(transport.post_calls) == 1
    assert "duration_control_enabled" not in transport.post_calls[0]["json"]
    assert result.outputs["tts_provider_retry_count"] == 0


def test_indextts_stage_runner_retries_low_occupancy_source_window_with_duration_control(tmp_path: Path) -> None:
    text = "one two three four five six seven eight nine ten"
    transport = DurationAwareTransport({text: 5.0})
    runner = TtsStageRunner(
        provider=IndexTtsProvider(transport),
        settings=TtsSettings(
            max_retries=1,
            provider_config={
                "api_url": "http://x/tts",
                "duration_control": {
                    "enabled": False,
                    "adaptive_source_window_retry": {
                        "enabled": True,
                        "screen_max_audio_speed": 1.07,
                        "max_window_gap_sec": 0.0,
                        "low_occupancy_retry_enabled": True,
                        "low_occupancy_min_window_sec": 3.0,
                        "low_occupancy_min_slack_sec": 1.0,
                        "low_occupancy_min_ratio": 0.70,
                        "low_occupancy_target_ratio": 0.75,
                        "low_occupancy_max_duration_factor": 1.50,
                        "low_occupancy_min_spoken_weight": 10.0,
                    },
                },
            },
            audio_config={"postprocess_audio": False},
        ),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "tts_segments": [
                    {
                        "id": "1_0",
                        "number": 1,
                        "start": 0.0,
                        "end": 10.0,
                        "text": text,
                        "output_path": "output/audio/tmp/1_0_temp.wav",
                    },
                ]
            },
            stage=StageName.TTS,
            attempt=1,
        )
    )

    assert len(transport.post_calls) == 2
    retry_payload = transport.post_calls[1]["json"]
    assert retry_payload["duration_control_enabled"] is True
    assert retry_payload["target_duration_sec"] == pytest.approx(7.5)
    assert result.outputs["tts_provider_retry_count"] == 1
    assert result.outputs["tts_durations"]["1_0"] == pytest.approx(7.5)

    report = json.loads(Path(result.outputs["tts_audio_quality_report"]).read_text(encoding="utf-8"))
    retry_info = report["segments"][0]["provider_retry"]
    assert retry_info["retry_kind"] == "indextts_low_occupancy_duration_control"
    assert retry_info["source_window_occupancy_ratio"] == pytest.approx(0.5)
    assert retry_info["kept"] is True


def _wav_bytes(duration_ms: int = 100, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    frame_count = max(1, round(sample_rate * duration_ms / 1000))
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for index in range(frame_count):
            value = int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate))
            handle.writeframes(struct.pack("<h", value))
    return buffer.getvalue()
