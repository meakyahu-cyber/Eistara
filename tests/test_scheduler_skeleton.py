from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import eistara.core.diagnostics.loader as diagnostics_loader
from eistara.core.jobs import JsonJobStore, JobStatus, StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.pipeline import StageContext, StageResult
from eistara.config import AppConfig
from eistara.core.scheduler import SchedulerDependencyProbe, SchedulerRecoveryPolicy, SchedulerService


class FakeDownloadRunner:
    stage = StageName.DOWNLOAD

    def run(self, context: StageContext) -> StageResult:
        return StageResult(outputs={"ok": True})


class FailedDownloadRunner:
    stage = StageName.DOWNLOAD

    def run(self, context: StageContext) -> StageResult:
        return StageResult(status="failed", outputs={"error": "quality gate failed"}, warnings=["quality gate failed"])


class RecordingRunner:
    def __init__(self, stage: StageName, calls: list[tuple[str, StageName]]):
        self.stage = stage
        self.calls = calls

    def run(self, context: StageContext) -> StageResult:
        self.calls.append((context.job_id, context.stage))
        return StageResult(outputs={f"{context.stage.value}_ok": True})


def write_job(
    jobs_dir: Path,
    status: str = "pending",
    current_stage: str | None = None,
    *,
    job_id: str = "job_0001_test",
    completed_stages: list[str] | None = None,
) -> Path:
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True)
    (job_dir / TASK_FILE).write_text(
        json.dumps({"id": job_dir.name, "source": "demo.mp4", "title": "Demo"}, indent=2),
        encoding="utf-8",
    )
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": status,
                "current_stage": current_stage,
                "completed_stages": completed_stages or [],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return job_dir


def write_failed_job(jobs_dir: Path) -> Path:
    job_dir = write_job(jobs_dir, status="failed")
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["failed_stage"] = "download"
    state["error"] = "failed once"
    (job_dir / STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    return job_dir


def old_timestamp() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(timespec="seconds")


def old_epoch() -> float:
    return (datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()


def test_recover_interrupted_returns_running_job_to_pending(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, status="running", current_stage="translate")

    recovered = JsonJobStore(jobs_dir).recover_interrupted()

    assert len(recovered) == 1
    assert recovered[0].state.status == JobStatus.PENDING
    assert recovered[0].state.current_stage is None


def test_job_store_reads_utf8_sig_state(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state_path = job_dir / STATE_FILE
    state_path.write_text("\ufeff" + state_path.read_text(encoding="utf-8"), encoding="utf-8")

    job = JsonJobStore(jobs_dir).load("job_0001_test")

    assert job.state.job_id == "job_0001_test"


def test_job_store_compacts_heavy_state_artifacts(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    segments_json = tmp_path / "jobs" / "job_0001_test" / "output" / "internal" / "tts_segments.json"
    JsonJobStore(jobs_dir).mark_done(
        "job_0001_test",
        StageName.TRANSLATE,
        {
            "tts_segments": [{"id": "1"}, {"id": "2"}],
            "tts_segments_json": str(segments_json),
            "translations": {1: "hello", 2: "world"},
            "small_value": "kept",
        },
    )

    state = json.loads((jobs_dir / "job_0001_test" / STATE_FILE).read_text(encoding="utf-8"))
    artifacts = state["artifacts"]
    assert artifacts["tts_segments_json"] == str(segments_json)
    assert artifacts["tts_segments_count"] == 2
    assert artifacts["translations_count"] == 2
    assert artifacts["small_value"] == "kept"
    assert "tts_segments" not in artifacts
    assert "translations" not in artifacts
    assert artifacts["_omitted_inline_artifacts"] == ["tts_segments", "translations"]


def test_scheduler_runs_registered_stage(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)

    service = SchedulerService(jobs_dir)
    service.register(FakeDownloadRunner())

    assert service.run_one_ready_stage() is True
    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.status == JobStatus.PENDING
    assert job.state.completed_stages == [StageName.DOWNLOAD]
    assert job.state.artifacts["ok"] is True


def test_scheduler_notifies_local_diagnostics_hook(tmp_path: Path, monkeypatch) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    hook_dir = tmp_path / "diagnostics"
    hook_dir.mkdir()
    (hook_dir / "diag_hook.py").write_text(
        "\n".join(
            [
                "import json",
                "",
                "def on_stage_finished(context, result):",
                "    (context.job_dir / 'diagnostic_event.json').write_text(",
                "        json.dumps({'job_id': context.job_id, 'stage': context.stage.value, 'status': result.status}),",
                "        encoding='utf-8',",
                "    )",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("EISTARA_DIAGNOSTICS_MODULE", "diag_hook")
    monkeypatch.setenv("EISTARA_DIAGNOSTICS_PATH", str(hook_dir))
    diagnostics_loader._HOOK = None

    try:
        service = SchedulerService(jobs_dir)
        service.register(FakeDownloadRunner())

        assert service.run_one_ready_stage() is True
    finally:
        diagnostics_loader._HOOK = None

    event = json.loads((job_dir / "diagnostic_event.json").read_text(encoding="utf-8"))
    assert event == {"job_id": "job_0001_test", "stage": "download", "status": "done"}


def test_scheduler_can_run_multiple_registered_stages(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)

    service = SchedulerService(jobs_dir)
    calls: list[tuple[str, StageName]] = []
    service.register(RecordingRunner(StageName.DOWNLOAD, calls))
    service.register(RecordingRunner(StageName.TRANSCRIBE, calls))

    assert service.run_one_ready_stage() is True
    assert service.run_one_ready_stage() is True
    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.completed_stages == [StageName.DOWNLOAD, StageName.TRANSCRIBE]
    assert calls == [("job_0001_test", StageName.DOWNLOAD), ("job_0001_test", StageName.TRANSCRIBE)]


def test_scheduler_matches_v1_stage_priority_before_job_order(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_download")
    write_job(jobs_dir, job_id="job_0002_translate", completed_stages=["download", "transcribe"])
    calls: list[tuple[str, StageName]] = []
    service = SchedulerService(jobs_dir)
    service.register(RecordingRunner(StageName.DOWNLOAD, calls))
    service.register(RecordingRunner(StageName.TRANSLATE, calls))

    assert service.run_one_ready_stage() is True

    assert calls == [("job_0002_translate", StageName.TRANSLATE)]


def test_scheduler_honors_stage_worker_limit_from_policy(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, status="running", current_stage="translate", job_id="job_0001_running")
    write_job(jobs_dir, job_id="job_0002_ready", completed_stages=["download", "transcribe"])
    calls: list[tuple[str, StageName]] = []
    service = SchedulerService(jobs_dir)
    service.register(RecordingRunner(StageName.TRANSLATE, calls))

    assert service.run_one_ready_stage() is False
    assert calls == []


def test_scheduler_holds_dependency_blocked_stage_pending(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_translate", completed_stages=["download", "transcribe"])
    calls: list[tuple[str, StageName]] = []
    config = AppConfig.from_dict({"api": {"base_url": "http://llm.test/v1"}})
    service = SchedulerService(
        jobs_dir,
        dependencies=SchedulerDependencyProbe.from_config(config, url_probe=lambda url, timeout: False),
    )
    service.register(RecordingRunner(StageName.TRANSLATE, calls))

    assert service.run_one_ready_stage() is False

    job = JsonJobStore(jobs_dir).load("job_0001_translate")
    assert job.state.status == JobStatus.PENDING
    assert job.state.attempts == {}
    assert calls == []


def test_scheduler_blocks_translate_when_llm_chat_probe_fails(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_translate", completed_stages=["download", "transcribe"])
    calls: list[tuple[str, StageName]] = []
    chat_calls: list[tuple[str, str | None, str | None, bool, float, str | None, bool]] = []
    config = AppConfig.from_dict(
        {"api": {"base_url": "http://llm.test/v1", "key": "secret", "model": "model-a", "llm_support_json": True}}
    )

    def chat_probe(
        base_url: str,
        api_key: str | None,
        model: str | None,
        llm_support_json: bool,
        timeout: float,
        proxy_url: str | None,
        trust_env_proxy: bool,
    ):
        chat_calls.append((base_url, api_key, model, llm_support_json, timeout, proxy_url, trust_env_proxy))
        return False, "chat failed"

    service = SchedulerService(
        jobs_dir,
        dependencies=SchedulerDependencyProbe.from_config(
            config,
            url_probe=lambda url, timeout: True,
            llm_chat_probe=chat_probe,
        ),
    )
    service.register(RecordingRunner(StageName.TRANSLATE, calls))

    assert service.run_one_ready_stage() is False

    job = JsonJobStore(jobs_dir).load("job_0001_translate")
    assert job.state.status == JobStatus.PENDING
    assert job.state.attempts == {}
    assert calls == []
    assert chat_calls == [("http://llm.test/v1", "secret", "model-a", True, 30.0, None, True)]


def test_scheduler_dependency_probe_prefers_runtime_config_over_stale_job_config(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir, job_id="job_0001_translate", completed_stages=["download", "transcribe"])
    (job_dir / "config.yaml").write_text(
        "api:\n  base_url: http://old.example/v1\n  key: old-key\n  model: old-model\n",
        encoding="utf-8",
    )
    chat_calls: list[tuple[str, str | None, str | None, bool, float, str | None, bool]] = []
    config = AppConfig.from_dict(
        {
            "api": {
                "base_url": "http://new.example/v1",
                "key": "new-key",
                "model": "new-model",
                "llm_support_json": True,
                "proxy_url": "http://proxy.test:7890",
            }
        }
    )

    def chat_probe(
        base_url: str,
        api_key: str | None,
        model: str | None,
        llm_support_json: bool,
        timeout: float,
        proxy_url: str | None,
        trust_env_proxy: bool,
    ):
        chat_calls.append((base_url, api_key, model, llm_support_json, timeout, proxy_url, trust_env_proxy))
        return False, "blocked"

    probe = SchedulerDependencyProbe.from_config(
        config,
        url_probe=lambda url, timeout: True,
        llm_chat_probe=chat_probe,
    )
    job = JsonJobStore(jobs_dir).load("job_0001_translate")

    ready, reason = probe.ready(job, StageName.TRANSLATE)

    assert ready is False
    assert "blocked" in reason
    assert chat_calls == [("http://new.example/v1", "new-key", "new-model", True, 30.0, "http://proxy.test:7890", True)]


def test_scheduler_skips_blocked_stage_and_runs_lower_priority_work(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_download")
    write_job(jobs_dir, job_id="job_0002_translate", completed_stages=["download", "transcribe"])
    calls: list[tuple[str, StageName]] = []
    config = AppConfig.from_dict({"api": {"base_url": "http://llm.test/v1"}})
    service = SchedulerService(
        jobs_dir,
        dependencies=SchedulerDependencyProbe.from_config(config, url_probe=lambda url, timeout: False),
    )
    service.register(RecordingRunner(StageName.DOWNLOAD, calls))
    service.register(RecordingRunner(StageName.TRANSLATE, calls))

    assert service.run_one_ready_stage() is True

    assert calls == [("job_0001_download", StageName.DOWNLOAD)]


def test_scheduler_does_not_skip_blocked_first_job_within_same_stage(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir, job_id="job_0001_translate", completed_stages=["download", "transcribe"])
    write_job(jobs_dir, job_id="job_0002_translate", completed_stages=["download", "transcribe"])
    calls: list[tuple[str, StageName]] = []

    class FirstJobBlockedDependency:
        def ready(self, job, stage):
            return job.job_id != "job_0001_translate", "blocked"

    service = SchedulerService(jobs_dir, dependencies=FirstJobBlockedDependency())
    service.register(RecordingRunner(StageName.TRANSLATE, calls))

    assert service.run_one_ready_stage() is False

    assert calls == []


def test_scheduler_marks_failed_stage_result_as_failed(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)

    service = SchedulerService(jobs_dir)
    service.register(FailedDownloadRunner())

    assert service.run_one_ready_stage() is True
    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.status == JobStatus.FAILED
    assert job.state.failed_stage == StageName.DOWNLOAD
    assert job.state.completed_stages == []
    assert job.state.error == "quality gate failed"


def test_scheduler_does_not_auto_run_failed_jobs(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_failed_job(jobs_dir)
    service = SchedulerService(jobs_dir)
    service.register(FakeDownloadRunner())

    assert service.run_one_ready_stage() is False
    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.status == JobStatus.FAILED
    assert job.state.failed_stage == StageName.DOWNLOAD


def test_scheduler_retry_failed_moves_job_to_pending(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_failed_job(jobs_dir)
    service = SchedulerService(jobs_dir)
    service.register(FakeDownloadRunner())

    result = service.retry_failed("job_0001_test")

    assert result == {"job_id": "job_0001_test", "status": "pending", "failed_stage": "download"}
    assert service.run_one_ready_stage() is True
    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.completed_stages == [StageName.DOWNLOAD]


def test_scheduler_auto_requeues_cooled_failed_job(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_failed_job(jobs_dir)
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["updated_at"] = old_timestamp()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    service = SchedulerService(
        jobs_dir,
        recovery=SchedulerRecoveryPolicy(auto_requeue_failed=True, failed_cooldown_sec=1, max_auto_requeues=2),
    )
    service.register(FakeDownloadRunner())

    assert service.run_one_ready_stage() is True

    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.auto_requeue_count == 1
    assert job.state.completed_stages == [StageName.DOWNLOAD]


def test_scheduler_stalled_running_job_retries_without_running_stage(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir, status="running", current_stage="download")
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["attempts"] = {"download": 1}
    state["stage_started_at"] = old_timestamp()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    old_ts = old_epoch()
    os.utime(state_path, (old_ts, old_ts))
    service = SchedulerService(
        jobs_dir,
        max_stage_retries=1,
        recovery=SchedulerRecoveryPolicy(stage_idle_timeouts={StageName.DOWNLOAD: 1}),
    )

    assert service.recover_stalled_jobs() == 1

    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.status == JobStatus.PENDING
    assert job.state.current_stage is None
    assert job.state.failed_stage == StageName.DOWNLOAD
    assert "stalled" in (job.state.error or "")


def test_scheduler_stalled_running_job_uses_v1_log_progress_mtime(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir, status="running", current_stage="download")
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["attempts"] = {"download": 1}
    state["stage_started_at"] = old_timestamp()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    old_ts = old_epoch()
    os.utime(state_path, (old_ts, old_ts))
    log_path = job_dir / "logs" / "download.log"
    log_path.parent.mkdir()
    log_path.write_text("still making progress\n", encoding="utf-8")
    service = SchedulerService(
        jobs_dir,
        max_stage_retries=1,
        recovery=SchedulerRecoveryPolicy(stage_idle_timeouts={StageName.DOWNLOAD: 1}),
    )

    assert service.recover_stalled_jobs() == 0

    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.status == JobStatus.RUNNING
    assert job.state.current_stage == StageName.DOWNLOAD


def test_scheduler_stalled_running_job_fails_after_retry_budget(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir, status="running", current_stage="download")
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["attempts"] = {"download": 1}
    state["stage_started_at"] = old_timestamp()
    state_path.write_text(json.dumps(state), encoding="utf-8")
    old_ts = old_epoch()
    os.utime(state_path, (old_ts, old_ts))
    service = SchedulerService(
        jobs_dir,
        max_stage_retries=0,
        recovery=SchedulerRecoveryPolicy(stage_idle_timeouts={StageName.DOWNLOAD: 1}),
    )

    assert service.recover_stalled_jobs() == 1

    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert job.state.status == JobStatus.FAILED
    assert job.state.failed_stage == StageName.DOWNLOAD
    assert "stalled" in (job.state.error or "")


def test_scheduler_reset_from_stage_clears_downstream_state(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["status"] = "done"
    state["completed_stages"] = ["download", "transcribe", "translate", "tts_prepare"]
    state["attempts"] = {"download": 1, "transcribe": 1, "translate": 2, "tts_prepare": 1}
    state["artifacts"] = {
        "source_video": "source.mp4",
        "subtitle_rows_json": "rows.json",
        "translations_json": "translations.json",
        "tts_segments": [{"id": "1"}],
    }
    (job_dir / STATE_FILE).write_text(json.dumps(state), encoding="utf-8")
    service = SchedulerService(jobs_dir)

    result = service.reset_from_stage("job_0001_test", "translate")

    job = JsonJobStore(jobs_dir).load("job_0001_test")
    assert result["completed_stages"] == ["download", "transcribe"]
    assert job.state.status == JobStatus.PENDING
    assert job.state.completed_stages == [StageName.DOWNLOAD, StageName.TRANSCRIBE]
    assert StageName.TRANSLATE not in job.state.attempts
    assert job.state.artifacts == {"source_video": "source.mp4", "subtitle_rows_json": "rows.json"}
