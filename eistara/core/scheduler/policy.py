from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping

from eistara.core.jobs import Job, JobState, JobStatus, STAGE_ORDER, StageName


DEFAULT_STAGE_PRIORITY: tuple[StageName, ...] = (
    StageName.TTS,
    StageName.TRANSLATE,
    StageName.TTS_PREPARE,
    StageName.AUDIO_MIX,
    StageName.COMPOSE,
    StageName.TRANSCRIBE,
    StageName.DOWNLOAD,
)

DEFAULT_STAGE_WORKER_LIMITS: dict[StageName, int] = {
    StageName.DOWNLOAD: 3,
    StageName.TRANSCRIBE: 1,
    StageName.TRANSLATE: 1,
    StageName.TTS_PREPARE: 1,
    StageName.TTS: 1,
    StageName.AUDIO_MIX: 1,
    StageName.COMPOSE: 1,
}


@dataclass(frozen=True, slots=True)
class SchedulerSelection:
    job: Job
    stage: StageName


@dataclass(frozen=True, slots=True)
class SchedulerPolicy:
    max_active_jobs: int = 10
    stage_worker_limits: dict[StageName, int] = field(default_factory=lambda: dict(DEFAULT_STAGE_WORKER_LIMITS))
    stage_priority: tuple[StageName, ...] = DEFAULT_STAGE_PRIORITY

    @classmethod
    def from_batch_config(cls, batch: object) -> "SchedulerPolicy":
        limits: dict[StageName, int] = dict(DEFAULT_STAGE_WORKER_LIMITS)
        raw_limits: Mapping[object, object] = {}
        if hasattr(batch, "stage_worker_limits"):
            raw_limits = getattr(batch, "stage_worker_limits")()
        for key, value in raw_limits.items():
            try:
                limits[StageName(str(key))] = _positive_int(value, limits.get(StageName(str(key)), 1))
            except ValueError:
                continue
        return cls(
            max_active_jobs=_positive_int(getattr(batch, "max_active_jobs", 10), 10),
            stage_worker_limits=limits,
        )

    def limit_for(self, stage: StageName) -> int:
        return _positive_int(self.stage_worker_limits.get(stage), DEFAULT_STAGE_WORKER_LIMITS.get(stage, 1))

    def select_ready_job(
        self,
        jobs: Iterable[Job],
        next_stage: Callable[[JobState], StageName | None],
        *,
        registered_stages: Iterable[StageName] | None = None,
        can_start: Callable[[Job, StageName], bool] | None = None,
    ) -> SchedulerSelection | None:
        jobs = list(jobs)
        running_jobs = [job for job in jobs if job.state.status == JobStatus.RUNNING]
        if len(running_jobs) >= self.max_active_jobs:
            return None

        registered = set(registered_stages if registered_stages is not None else STAGE_ORDER)
        running_by_stage: Counter[StageName] = Counter(
            job.state.current_stage for job in running_jobs if job.state.current_stage is not None
        )
        ready_by_stage: dict[StageName, list[Job]] = {stage: [] for stage in STAGE_ORDER}
        for stage in registered:
            ready_by_stage.setdefault(stage, [])
        for stage in self.stage_priority:
            ready_by_stage.setdefault(stage, [])
        for job in jobs:
            if job.state.status != JobStatus.PENDING:
                continue
            stage = next_stage(job.state)
            if stage is None or stage not in registered:
                continue
            ready_by_stage.setdefault(stage, []).append(job)

        for stage in self.stage_priority:
            if stage not in registered:
                continue
            if running_by_stage[stage] >= self.limit_for(stage):
                continue
            ready_jobs = ready_by_stage.get(stage) or []
            if not ready_jobs:
                continue
            job = ready_jobs[0]
            if can_start is not None and not can_start(job, stage):
                continue
            return SchedulerSelection(job, stage)
        return None


def _positive_int(value: object, default: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default
