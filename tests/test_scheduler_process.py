from __future__ import annotations

import json
from pathlib import Path

from eistara.config import AppConfig, ConfigLoader
from eistara.core.jobs import JobStatus, JsonJobStore, StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
import eistara.core.scheduler.process as process_module
from eistara.core.scheduler.worker_result import read_stage_worker_result
from eistara.runtime import build_process_supervisor
from eistara.runtime.worker import run_stage_worker
from apps.cli.main import _use_process_scheduler


def write_job(
    jobs_dir: Path,
    *,
    job_id: str = "job_0001_process",
    completed_stages: list[str] | None = None,
    task: dict | None = None,
) -> Path:
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / TASK_FILE).write_text(
        json.dumps({"id": job_dir.name, "source": "demo.mp4", **(task or {})}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": completed_stages or [],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return job_dir


def test_stage_worker_runs_one_production_download_stage_without_mutating_state(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    write_job(jobs_dir, task={"source": str(source), "source_type": "file"})
    store = JsonJobStore(jobs_dir)
    store.mark_running("job_0001_process", StageName.DOWNLOAD, stage_run_token="token-1")
    result_path = tmp_path / "result.json"

    exit_code = run_stage_worker(
        jobs_dir=jobs_dir,
        job_id="job_0001_process",
        stage=StageName.DOWNLOAD,
        preset="production",
        result_path=result_path,
        run_token="token-1",
    )

    worker_result = read_stage_worker_result(result_path)
    job = store.load("job_0001_process")
    assert exit_code == 0
    assert worker_result.outputs["source_type"] == "file"
    assert Path(worker_result.outputs["source_video"]).exists()
    assert job.state.status == JobStatus.RUNNING
    assert job.state.stage_run_token == "token-1"


def test_process_supervisor_launches_available_jobs_and_clears_process_state(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    write_job(jobs_dir, job_id="job_0001_download", task={"source": str(source), "source_type": "file"})
    write_job(jobs_dir, job_id="job_0002_download", task={"source": str(source), "source_type": "file"})
    config = AppConfig.from_dict({"batch": {"max_active_jobs": 2, "download_workers": 2, "dependency_probe": False}})
    supervisor = build_process_supervisor(jobs_dir, preset="production", config=config)

    tick = supervisor.run_once(launch=True)

    assert tick.launched == 2
    running = [JsonJobStore(jobs_dir).load("job_0001_download"), JsonJobStore(jobs_dir).load("job_0002_download")]
    assert {job.state.status for job in running} == {JobStatus.RUNNING}
    assert all(job.state.stage_pid for job in running)
    assert all(job.state.stage_run_token for job in running)

    drained = supervisor.wait_for_active(poll_interval=0.05, timeout_sec=10)

    assert drained.finished == 2
    for job_id in ("job_0001_download", "job_0002_download"):
        job = JsonJobStore(jobs_dir).load(job_id)
        assert job.state.status == JobStatus.PENDING
        assert job.state.completed_stages == [StageName.DOWNLOAD]
        assert job.state.stage_pid is None
        assert job.state.stage_run_token is None


def test_process_supervisor_fills_v1_pipeline_lanes_in_one_tick(tmp_path: Path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_tts_prepare", completed_stages=["download", "transcribe", "translate"])
    write_job(jobs_dir, job_id="job_0002_translate", completed_stages=["download", "transcribe"])
    config = AppConfig.from_dict(
        {
            "batch": {
                "max_active_jobs": 10,
                "translate_workers": 1,
                "tts_prepare_workers": 1,
                "dependency_probe": False,
            }
        }
    )

    class FakeProcess:
        _pid = 20000

        def __init__(self, *args, **kwargs) -> None:
            type(self)._pid += 1
            self.pid = type(self)._pid

        def poll(self):
            return None

        def terminate(self) -> None:
            pass

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            pass

    monkeypatch.setattr(process_module.subprocess, "Popen", FakeProcess)
    supervisor = build_process_supervisor(jobs_dir, preset="production", config=config)

    tick = supervisor.run_once(launch=True)

    assert tick.launched == 2
    running = {
        JsonJobStore(jobs_dir).load("job_0001_tts_prepare").state.current_stage,
        JsonJobStore(jobs_dir).load("job_0002_translate").state.current_stage,
    }
    assert running == {StageName.TTS_PREPARE, StageName.TRANSLATE}
    supervisor.terminate_all()


def test_process_supervisor_starts_worker_with_unbuffered_logs(tmp_path: Path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_download")
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 12345

        def __init__(self, *args, **kwargs) -> None:
            captured["env"] = kwargs["env"]

        def poll(self):
            return None

        def terminate(self) -> None:
            captured["terminated"] = True

        def wait(self, timeout=None):
            return 0

        def kill(self) -> None:
            captured["killed"] = True

    monkeypatch.setattr(process_module.subprocess, "Popen", FakeProcess)
    supervisor = build_process_supervisor(jobs_dir, preset="production", config=AppConfig.from_dict({"batch": {"dependency_probe": False}}))

    active = supervisor.launch_stage(JsonJobStore(jobs_dir).load("job_0001_download"), StageName.DOWNLOAD)

    assert active is not None
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["PYTHONUNBUFFERED"] == "1"
    supervisor.terminate_all()


def test_cli_production_run_loop_defaults_to_v1_process_scheduler() -> None:
    class Args:
        command = "run-loop"
        preset = "production"
        processes = False
        in_process = False

    assert _use_process_scheduler(Args()) is True


def test_cli_in_process_flag_disables_process_scheduler() -> None:
    class Args:
        command = "run-loop"
        preset = "production"
        processes = False
        in_process = True

    assert _use_process_scheduler(Args()) is False


def test_process_supervisor_applies_worker_exception_to_retry_budget(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    missing_source = tmp_path / "missing.mp4"
    write_job(jobs_dir, task={"source": str(missing_source), "source_type": "file"})
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "batch": {"dependency_probe": False, "max_stage_retries": 0},
                "api": {"base_url": "http://llm/v1", "model": "m"},
                "asr": {"provider": "whisper", "model": "tiny"},
                "demucs": False,
                "tts_method": "indextts",
            }
        ),
        encoding="utf-8",
    )
    config = ConfigLoader(config_path).load()
    supervisor = build_process_supervisor(
        jobs_dir,
        preset="production",
        config=config,
        config_path=config_path,
        max_stage_retries=0,
    )

    tick = supervisor.run_once(launch=True)
    drained = supervisor.wait_for_active(poll_interval=0.05, timeout_sec=10)

    job = JsonJobStore(jobs_dir).load("job_0001_process")
    assert tick.launched == 1
    assert drained.finished == 1
    assert job.state.status == JobStatus.FAILED
    assert job.state.failed_stage == StageName.DOWNLOAD
    assert "Source file not found" in (job.state.error or "")
    assert job.state.stage_pid is None
    assert job.state.stage_run_token is None
