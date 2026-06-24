from __future__ import annotations

import json
from pathlib import Path

from eistara.core.jobs import JobStatus, StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.observability import JobEvent, JobEventType, JsonlEventStore
from eistara.core.pipeline import StageContext, StageResult
from eistara.core.scheduler import SchedulerService


class SuccessfulRunner:
    stage = StageName.DOWNLOAD

    def run(self, context: StageContext) -> StageResult:
        return StageResult(outputs={"ok": True})


class FailingRunner:
    stage = StageName.DOWNLOAD

    def run(self, context: StageContext) -> StageResult:
        raise RuntimeError("boom")


def write_job(jobs_dir: Path) -> Path:
    job_dir = jobs_dir / "job_0001_events"
    job_dir.mkdir(parents=True)
    (job_dir / TASK_FILE).write_text(json.dumps({"id": job_dir.name}), encoding="utf-8")
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


def test_jsonl_event_store_roundtrip(tmp_path: Path) -> None:
    store = JsonlEventStore(tmp_path / "jobs")
    event = JobEvent("job_1", JobEventType.STAGE_STARTED, stage=StageName.DOWNLOAD, status="running", attempt=1)

    path = store.append(event)
    events = store.read_job("job_1")

    assert path.name == "events.jsonl"
    assert events[0].event_type == JobEventType.STAGE_STARTED
    assert events[0].stage == StageName.DOWNLOAD


def test_scheduler_records_success_events(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    service = SchedulerService(jobs_dir)
    service.register(SuccessfulRunner())

    assert service.run_one_ready_stage() is True

    events = JsonlEventStore(jobs_dir).read_job("job_0001_events")
    assert [event.event_type for event in events] == [JobEventType.STAGE_STARTED, JobEventType.STAGE_FINISHED]
    assert events[-1].outputs == {"ok": True}
    assert events[-1].duration_sec is not None


def test_scheduler_records_retry_event(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    service = SchedulerService(jobs_dir)
    service.max_stage_retries = 1
    service.register(FailingRunner())

    service.run_one_ready_stage()

    events = JsonlEventStore(jobs_dir).read_job("job_0001_events")
    assert events[-1].event_type == JobEventType.STAGE_RETRYING
    assert "boom" in (events[-1].error or "")


def test_scheduler_records_failed_event_after_retries(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    service = SchedulerService(jobs_dir)
    service.max_stage_retries = 0
    service.register(FailingRunner())

    service.run_one_ready_stage()

    events = JsonlEventStore(jobs_dir).read_job("job_0001_events")
    assert events[-1].event_type == JobEventType.STAGE_FAILED
    state = json.loads((jobs_dir / "job_0001_events" / STATE_FILE).read_text(encoding="utf-8"))
    assert state["status"] == JobStatus.FAILED.value


def test_scheduler_records_retry_requested_event(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    state_path = jobs_dir / "job_0001_events" / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "failed"
    state["failed_stage"] = "download"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    service = SchedulerService(jobs_dir)

    service.retry_failed("job_0001_events")

    events = JsonlEventStore(jobs_dir).read_job("job_0001_events")
    assert events[-1].event_type == JobEventType.JOB_RETRY_REQUESTED
    assert events[-1].stage == StageName.DOWNLOAD


def test_scheduler_records_reset_requested_event(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    service = SchedulerService(jobs_dir)

    service.reset_from_stage("job_0001_events", "download")

    events = JsonlEventStore(jobs_dir).read_job("job_0001_events")
    assert events[-1].event_type == JobEventType.JOB_RESET_REQUESTED
    assert events[-1].stage == StageName.DOWNLOAD
