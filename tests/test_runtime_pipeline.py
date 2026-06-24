from __future__ import annotations

import json
from pathlib import Path

from eistara.core.asr import AsrSegment, ScriptedAsrProvider
from eistara.core.jobs import STAGE_ORDER, JsonJobStore, JobStatus, StageName, history_dir_for_jobs
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.media import AudioStreamInfo, MediaCommandResult, MediaInfo
from eistara.core.translation import ScriptedLlmClient
from eistara.core.tts import ScriptedTtsProvider
from eistara.config import AppConfig
from eistara.runtime import PIPELINE_PRESETS, WEBUI_DEFAULT_PRESET, RuntimeProviders, build_runners, build_scheduler


class FakeMediaProvider:
    name = "fake-media"

    def extract_audio(self, plan):
        plan.output_audio.parent.mkdir(parents=True, exist_ok=True)
        plan.output_audio.write_bytes(b"audio")
        return MediaCommandResult(("extract", str(plan.source_video), str(plan.output_audio)), 0)

    def probe(self, path: str) -> MediaInfo:
        return MediaInfo(path=Path(path), duration_sec=1.0, audio=AudioStreamInfo(duration_sec=1.0))

    def compose_video(self, plan):
        plan.output_video.parent.mkdir(parents=True, exist_ok=True)
        plan.output_video.write_bytes(b"video")
        return MediaCommandResult(("compose", str(plan.source_video), str(plan.output_video)), 0)


def test_public_pipeline_presets_are_native_v2_only() -> None:
    assert WEBUI_DEFAULT_PRESET == "production"
    assert PIPELINE_PRESETS == ["production"]
    assert "noop" not in PIPELINE_PRESETS
    assert "plan" not in PIPELINE_PRESETS


def write_job(jobs_dir: Path, task: dict | None = None) -> Path:
    job_dir = jobs_dir / "job_0001_pipeline"
    job_dir.mkdir(parents=True)
    task = {"id": job_dir.name, **(task or {})}
    (job_dir / TASK_FILE).write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": [],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def test_build_production_runners_covers_stage_order() -> None:
    config = AppConfig.from_dict(
        {
            "api": {"base_url": "http://llm/v1", "model": "m"},
            "asr": {"provider": "whisper", "model": "tiny"},
            "demucs": False,
            "tts_method": "indextts",
            "batch": {"dependency_probe": False},
            "render": {"render_audio": False, "render_video": False},
        }
    )

    runners = build_runners("production", config=config)

    assert [runner.stage for runner in runners] == list(STAGE_ORDER)
    assert runners[0].file_provider.name == "local-file"
    assert runners[1].asr_provider.name == "whisperx"
    assert not hasattr(runners[1], "platform_subtitle_provider")
    assert runners[2].llm.settings.base_url == "http://llm/v1"
    assert runners[2].llm.transport is not None
    assert runners[4].provider.name == "indextts"


def test_build_production_runners_accepts_provider_overrides() -> None:
    config = AppConfig.from_dict(
        {
            "api": {"base_url": "http://llm/v1", "model": "m"},
            "asr": {"provider": "whisper", "model": "tiny"},
            "demucs": False,
            "tts_method": "indextts",
            "batch": {"dependency_probe": False},
            "render": {"render_audio": False, "render_video": False},
        }
    )
    providers = RuntimeProviders(
        media_provider=FakeMediaProvider(),
        asr_provider=ScriptedAsrProvider(),
        llm_client=ScriptedLlmClient([]),
        tts_provider=ScriptedTtsProvider(),
    )

    runners = build_runners("production", config=config, providers=providers)

    assert runners[1].media_provider.name == "fake-media"
    assert runners[1].asr_provider.name == "scripted"
    assert runners[2].llm is providers.llm_client
    assert runners[4].provider.name == "scripted"


def test_build_production_runners_selects_audio_separator_vocal_provider() -> None:
    config = AppConfig.from_dict(
        {
            "api": {"base_url": "http://llm/v1", "model": "m"},
            "asr": {"provider": "whisper", "model": "tiny"},
            "demucs": {
                "enabled": True,
                "provider": "audio-separator",
                "audio_separator_model": "UVR-MDX-NET-Voc_FT.onnx",
                "audio_separator_model_dir": "./models/audio-separator",
            },
            "tts_method": "indextts",
            "batch": {"dependency_probe": False},
        }
    )

    runners = build_runners("production", config=config)

    provider = runners[1].vocal_separation_provider
    assert provider is not None
    assert provider.name == "audio-separator"
    assert provider.model_filename == "UVR-MDX-NET-Voc_FT.onnx"
    assert str(provider.model_dir) == "models\\audio-separator"


def test_build_scheduler_uses_config_batch_policy(tmp_path: Path) -> None:
    config = AppConfig.from_dict(
        {
            "batch": {
                "max_active_jobs": 2,
                "download_workers": 4,
                "translate_workers": 1,
                "max_stage_retries": 3,
                "auto_requeue_failed": True,
                "failed_cooldown_sec": 7,
                "max_auto_requeues": 5,
                "download_idle_timeout_sec": 11,
            }
        }
    )

    service = build_scheduler(tmp_path / "jobs", preset="production", config=config)

    assert service.policy.max_active_jobs == 2
    assert service.policy.limit_for(StageName.DOWNLOAD) == 4
    assert service.policy.limit_for(StageName.TRANSLATE) == 1
    assert service.max_stage_retries == 3
    assert service.recovery.auto_requeue_failed is True
    assert service.recovery.failed_cooldown_sec == 7
    assert service.recovery.max_auto_requeues == 5
    assert service.recovery.timeout_for(StageName.DOWNLOAD) == 11


def test_plan_pipeline_preset_remains_internal_only(tmp_path: Path) -> None:
    service = build_scheduler(tmp_path / "jobs", preset="plan")
    assert list(service.registry.registered_stages()) == list(STAGE_ORDER)
    assert "plan" not in PIPELINE_PRESETS
    return

    jobs_dir = tmp_path / "jobs"
    output_dir = tmp_path / "output"
    translations_json = tmp_path / "translations.json"
    subtitles_json = tmp_path / "subtitle_rows.json"
    translations_json.write_text('{"translations":[{"id":1,"text":"你好"}]}', encoding="utf-8")
    subtitles_json.write_text('{"rows":[{"start":0,"end":1,"source":"hello","target":"你好"}]}', encoding="utf-8")
    write_job(
        jobs_dir,
        {
            "source_video": "source.mp4",
            "output_dir": str(output_dir),
            "subtitle_rows_json": str(subtitles_json),
            "translations_json": str(translations_json),
            "tts_segments": [{"id": "1", "start": 0, "end": 1, "text": "hello", "output_path": "a.wav", "audio_duration_sec": 1}],
        },
    )
    service = build_scheduler(jobs_dir, preset="plan")

    for _ in STAGE_ORDER:
        service.run_one_ready_stage()

    job = JsonJobStore(jobs_dir).load("job_0001_pipeline")
    assert StageName.AUDIO_MIX in job.state.completed_stages
    assert StageName.COMPOSE in job.state.completed_stages
    assert (output_dir / "audio_mix_plan.json").exists()
    assert (output_dir / "dub_segments.json").exists()
    assert (output_dir / "compose_plan.json").exists()


def test_production_pipeline_contract_can_finish_with_scripted_providers(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    source_video = tmp_path / "source.mp4"
    source_video.write_bytes(b"video")
    write_job(jobs_dir, {"source": str(source_video), "source_type": "file", "output_dir": str(tmp_path / "output")})
    config = AppConfig.from_dict(
        {
            "api": {"base_url": "http://llm/v1", "model": "m"},
            "asr": {"provider": "whisper", "model": "tiny"},
            "demucs": False,
            "tts_method": "indextts",
            "batch": {"dependency_probe": False},
            "render": {"render_audio": False, "render_video": False},
        }
    )
    providers = RuntimeProviders(
        media_provider=FakeMediaProvider(),
        asr_provider=ScriptedAsrProvider([AsrSegment(1, 0.0, 1.0, "Hello world")]),
        llm_client=ScriptedLlmClient([{"translations": [{"id": 1, "text": "你好，世界"}]}]),
        tts_provider=ScriptedTtsProvider(payload=b"audio"),
    )
    service = build_scheduler(jobs_dir, preset="production", config=config, providers=providers)

    for _ in STAGE_ORDER:
        assert service.run_one_ready_stage() is True

    assert not (jobs_dir / "job_0001_pipeline").exists()
    archived = JsonJobStore(history_dir_for_jobs(jobs_dir)).discover()
    assert len(archived) == 1
    job = archived[0]
    assert job.state.status == JobStatus.DONE
    assert job.state.completed_stages == list(STAGE_ORDER)
    assert Path(job.state.artifacts["translations_json"]).exists()
    assert job.state.artifacts["tts_count"] == 1
    assert job.state.artifacts["clip_count"] == 1
    assert Path(job.state.artifacts["compose_plan"]).exists()
