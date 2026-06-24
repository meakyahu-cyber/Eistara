from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from eistara.core.jobs import JsonJobStore, STAGE_ORDER, history_dir_for_jobs
from eistara.core.jobs.models import JobStatus, StageName
from eistara.core.manifest import JsonManifestStore

from .heartbeat import SchedulerHeartbeat
from .lock import LOCK_FILE, is_pid_running, read_lock_pid


def collect_status_rows(jobs_dir: str | Path, *, include_history: bool = True) -> list[dict[str, Any]]:
    active_store = JsonJobStore(jobs_dir)
    stores = [(active_store, "jobs")]
    if include_history:
        stores.append((JsonJobStore(history_dir_for_jobs(jobs_dir)), "history"))
    manifests = JsonManifestStore()
    rows: list[dict[str, Any]] = []
    for store, location in stores:
        for job in store.discover():
            stage = job.state.current_stage or job.state.failed_stage or store.next_stage(job.state)
            manifest = manifests.load_or_create(job.job_dir, job.task)
            task_source_type = str(job.task.get("source_type") or "").strip().lower()
            artifact_source_type = str(
                job.state.artifacts.get("source_type")
                or manifest.outputs.get("source_type")
                or ""
            ).strip().lower()
            per_stage = {item.value: manifest.stages[item].status for item in manifest.stage_order}
            for completed_stage in job.state.completed_stages:
                per_stage[completed_stage.value] = "done"
            if job.state.current_stage:
                per_stage[job.state.current_stage.value] = "running"
            if job.state.failed_stage and job.state.status == JobStatus.FAILED:
                per_stage[job.state.failed_stage.value] = "failed"
            error = job.state.error or ""
            rows.append(
                {
                    "job": job.job_id,
                    "status": job.state.status.value,
                    "stage": stage.value if isinstance(stage, StageName) else "-",
                    "completed": ",".join(item.value for item in job.state.completed_stages),
                    "progress": f"{len(job.state.completed_stages)}/{len(STAGE_ORDER)}",
                    "updated": job.state.updated_at,
                    "source": str(job.task.get("title") or job.task.get("source") or ""),
                    "task_source": str(job.task.get("source") or ""),
                    "task_source_type": task_source_type,
                    "artifact_source_type": artifact_source_type,
                    "source_type_mismatch": bool(task_source_type and artifact_source_type and task_source_type != artifact_source_type),
                    "job_dir": str(job.job_dir),
                    "location": location,
                    "error": error,
                    "error_summary": _short_status_text(error),
                    "caption_source": manifest.caption_source or "",
                    "stage_statuses": per_stage,
                    "manifest": str(job.job_dir / "manifest.json"),
                }
            )
    return rows


def _short_status_text(value: str, limit: int = 220) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "sk-***REDACTED***", text)
    text = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}", r"\1***REDACTED***", text)
    text = re.sub(r"(?i)(api[_-]?key|token|secret|password|authorization|cookie)(\s*[:=]\s*)([^\s,;]+)", r"\1\2***REDACTED***", text)
    for marker in (
        "Output from ffmpeg/avlib:",
        "ffmpeg version ",
        "Traceback (most recent call last):",
        "\n[in#",
        "\nlibavutil ",
    ):
        index = text.find(marker)
        if index >= 0:
            text = text[:index].strip() or "[details in diagnostic package]"
            break
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}... [truncated]"


def scheduler_health(jobs_dir: str | Path) -> dict[str, Any]:
    root = Path(jobs_dir).expanduser().resolve()
    heartbeat = SchedulerHeartbeat(root)
    hb = heartbeat.read()
    hb_age = heartbeat.age_sec()
    lock_path = root / LOCK_FILE
    lock_pid = read_lock_pid(lock_path)
    lock_pid_alive = is_pid_running(lock_pid)
    jobs = JsonJobStore(root).discover()
    running_jobs = [
        {
            "job": job.job_id,
            "stage": job.state.current_stage.value if job.state.current_stage else "",
            "started_at": job.state.stage_started_at,
        }
        for job in jobs
        if job.state.status == JobStatus.RUNNING
    ]
    stale_lock = lock_path.exists() and not lock_pid_alive
    stale_heartbeat = bool(hb) and not is_pid_running(_heartbeat_pid(hb))
    return {
        "jobs_dir": str(root),
        "heartbeat": hb,
        "heartbeat_age_sec": hb_age,
        "lock_present": lock_path.exists(),
        "lock_pid": lock_pid,
        "lock_pid_alive": lock_pid_alive,
        "stale_lock": stale_lock,
        "stale_heartbeat": stale_heartbeat,
        "running_jobs": running_jobs,
        "running_count": len(running_jobs),
        "needs_recovery": stale_lock or stale_heartbeat or bool(running_jobs and not lock_pid_alive),
    }


def _heartbeat_pid(heartbeat: dict[str, Any] | None) -> int | None:
    if not heartbeat:
        return None
    try:
        return int(heartbeat.get("pid"))
    except (TypeError, ValueError):
        return None
