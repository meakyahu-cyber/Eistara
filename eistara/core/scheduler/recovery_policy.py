from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from eistara.core.jobs import Job, STAGE_ORDER, StageName
from eistara.core.jobs.models import JobStatus


@dataclass(frozen=True, slots=True)
class SchedulerRecoveryPolicy:
    auto_requeue_failed: bool = False
    failed_cooldown_sec: int = 300
    max_auto_requeues: int = 0
    stage_idle_timeouts: dict[StageName, int] = field(default_factory=dict)

    @classmethod
    def from_batch_config(cls, batch: object) -> "SchedulerRecoveryPolicy":
        timeouts: dict[StageName, int] = {}
        raw_timeouts: Mapping[object, object] = {}
        if hasattr(batch, "stage_idle_timeouts"):
            raw_timeouts = getattr(batch, "stage_idle_timeouts")()
        for key, value in raw_timeouts.items():
            try:
                timeouts[StageName(str(key))] = max(0, int(value))
            except (TypeError, ValueError):
                continue
        generic_timeout = _non_negative_int(getattr(batch, "stage_idle_timeout_sec", 0), 0)
        if generic_timeout > 0:
            for stage in STAGE_ORDER:
                timeouts.setdefault(stage, generic_timeout)
        return cls(
            auto_requeue_failed=bool(getattr(batch, "auto_requeue_failed", False)),
            failed_cooldown_sec=_non_negative_int(getattr(batch, "failed_cooldown_sec", 300), 300),
            max_auto_requeues=_non_negative_int(getattr(batch, "max_auto_requeues", 0), 0),
            stage_idle_timeouts=timeouts,
        )

    def timeout_for(self, stage: StageName) -> int:
        return max(0, int(self.stage_idle_timeouts.get(stage, 0) or 0))

    def is_stalled(self, job: Job, *, now_ts: float | None = None) -> bool:
        if job.state.status != JobStatus.RUNNING or job.state.current_stage is None:
            return False
        timeout = self.timeout_for(job.state.current_stage)
        if timeout <= 0:
            return False
        progress_ts = _last_progress_ts(job)
        if progress_ts is None:
            return False
        return (_now_ts() if now_ts is None else now_ts) - progress_ts >= timeout

    def should_auto_requeue(self, job: Job, *, now_ts: float | None = None) -> bool:
        if not self.auto_requeue_failed or self.max_auto_requeues <= 0:
            return False
        if job.state.status != JobStatus.FAILED:
            return False
        if job.state.auto_requeue_count >= self.max_auto_requeues:
            return False
        updated_ts = _timestamp(job.state.updated_at)
        if updated_ts is None:
            return True
        return (_now_ts() if now_ts is None else now_ts) - updated_ts >= self.failed_cooldown_sec


def _timestamp(value: object) -> float | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


def _last_progress_ts(job: Job) -> float | None:
    latest = _timestamp(job.state.stage_started_at) or _timestamp(job.state.updated_at)
    stage = job.state.current_stage
    if stage is None:
        return latest
    for path in (
        job.job_dir / "logs" / f"{stage.value}.log",
        job.job_dir / "state.json",
    ):
        file_ts = _mtime(path)
        if file_ts is not None:
            latest = max(latest or file_ts, file_ts)
    if stage == StageName.TTS:
        latest = _latest_child_mtime(job.job_dir / "output" / "audio" / "tmp", latest)
    return latest


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _latest_child_mtime(path: Path, latest: float | None = None) -> float | None:
    try:
        children = list(path.iterdir())
    except OSError:
        return latest
    for child in children:
        file_ts = _mtime(child)
        if file_ts is not None:
            latest = max(latest or file_ts, file_ts)
    return latest


def _non_negative_int(value: object, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default
