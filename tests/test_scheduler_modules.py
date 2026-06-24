from __future__ import annotations

import json
from pathlib import Path

from eistara.core.jobs import JobFactory
from eistara.core.jobs.store import STATE_FILE
from eistara.core.scheduler.heartbeat import HEARTBEAT_FILE, SchedulerHeartbeat
from eistara.core.scheduler.lock import LOCK_FILE, SchedulerLock
from eistara.core.scheduler.recovery import recover_orphaned_scheduler_state
from eistara.core.scheduler.status import collect_status_rows, scheduler_health


def test_job_factory_creates_url_job_and_manifest(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("https://example.com/video\n", encoding="utf-8")

    created = JobFactory(tmp_path / "jobs", project_root=tmp_path).create_from_file(tasks)

    assert len(created) == 1
    assert (created[0] / "task.json").exists()
    assert (created[0] / "state.json").exists()
    assert (created[0] / "manifest.json").exists()


def test_job_factory_copies_config_template(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("https://example.com/video\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("target_language: Simplified Chinese\n", encoding="utf-8")

    created = JobFactory(tmp_path / "jobs", project_root=tmp_path, config_path=config).create_from_file(tasks)

    assert (created[0] / "config.yaml").read_text(encoding="utf-8") == "target_language: Simplified Chinese\n"


def test_job_factory_names_active_jobs_with_timestamp(tmp_path: Path) -> None:
    source = tmp_path / "示例视频.mp4"
    source.write_bytes(b"video")
    tasks = tmp_path / "tasks.txt"
    tasks.write_text(str(source), encoding="utf-8")

    created = JobFactory(tmp_path / "jobs", project_root=tmp_path).create_from_file(tasks)

    assert len(created[0].name) == 10
    assert created[0].name.isdigit()


def test_scheduler_lock_removes_stale_lock(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    lock_path = jobs_dir / LOCK_FILE
    lock_path.write_text("pid=999999\n", encoding="utf-8")

    with SchedulerLock(jobs_dir):
        assert lock_path.exists()

    assert not lock_path.exists()


def test_heartbeat_write_read_clear(tmp_path: Path) -> None:
    heartbeat = SchedulerHeartbeat(tmp_path / "jobs")
    heartbeat.write(active=["job_1:translate"])

    data = heartbeat.read()

    assert data is not None
    assert data["active"] == ["job_1:translate"]
    assert (tmp_path / "jobs" / HEARTBEAT_FILE).exists()
    heartbeat.clear()
    assert heartbeat.read() is None


def test_recovery_clears_stale_scheduler_files(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("https://example.com/video\n", encoding="utf-8")
    job_dir = JobFactory(tmp_path / "jobs", project_root=tmp_path).create_from_file(tasks)[0]
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["current_stage"] = "translate"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "jobs" / LOCK_FILE).write_text("pid=999999\n", encoding="utf-8")
    (tmp_path / "jobs" / HEARTBEAT_FILE).write_text(json.dumps({"pid": 999999}), encoding="utf-8")

    recovered = recover_orphaned_scheduler_state(tmp_path / "jobs")

    assert recovered == 1
    assert not (tmp_path / "jobs" / LOCK_FILE).exists()
    assert not (tmp_path / "jobs" / HEARTBEAT_FILE).exists()
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "pending"
    assert state["current_stage"] is None


def test_status_and_health_are_structured(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("https://example.com/video\n", encoding="utf-8")
    JobFactory(tmp_path / "jobs", project_root=tmp_path).create_from_file(tasks)

    rows = collect_status_rows(tmp_path / "jobs")
    health = scheduler_health(tmp_path / "jobs")

    assert rows[0]["status"] == "pending"
    assert rows[0]["stage"] == "download"
    assert health["lock_present"] is False
    assert health["running_count"] == 0
    assert health["needs_recovery"] is False


def test_status_exposes_source_type_mismatch(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("https://example.com/video\n", encoding="utf-8")
    job_dir = JobFactory(tmp_path / "jobs", project_root=tmp_path).create_from_file(tasks)[0]
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifacts"]["source_type"] = "file"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    rows = collect_status_rows(tmp_path / "jobs")

    assert rows[0]["task_source"] == "https://example.com/video"
    assert rows[0]["task_source_type"] == "url"
    assert rows[0]["artifact_source_type"] == "file"
    assert rows[0]["source_type_mismatch"] is True


def test_scheduler_health_reports_recovery_needed_for_running_without_live_lock(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("https://example.com/video\n", encoding="utf-8")
    job_dir = JobFactory(tmp_path / "jobs", project_root=tmp_path).create_from_file(tasks)[0]
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["current_stage"] = "translate"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (tmp_path / "jobs" / LOCK_FILE).write_text("pid=999999\n", encoding="utf-8")

    health = scheduler_health(tmp_path / "jobs")

    assert health["stale_lock"] is True
    assert health["running_count"] == 1
    assert health["running_jobs"][0]["stage"] == "translate"
    assert health["needs_recovery"] is True
