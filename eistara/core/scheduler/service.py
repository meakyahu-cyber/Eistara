from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from eistara.core.jobs import JsonJobStore, JobStatus, StageName
from eistara.core.jobs.models import Job
from eistara.core.manifest import JsonManifestStore
from eistara.core.observability import JobEvent, JobEventType, JsonlEventStore
from eistara.core.pipeline import StageContext, StageRegistry, StageResult, StageRunner

from .dependencies import SchedulerDependencyProbe
from .heartbeat import SchedulerHeartbeat
from .lock import SchedulerLock
from .policy import SchedulerPolicy, SchedulerSelection
from .recovery_policy import SchedulerRecoveryPolicy


@dataclass(slots=True)
class SchedulerService:
    """Small in-process scheduler facade.

    Eistara keeps this service focused on state
    transitions and delegates actual stage work to registered runners.
    """

    jobs_dir: Path
    registry: StageRegistry = field(default_factory=StageRegistry)
    max_stage_retries: int = 1
    policy: SchedulerPolicy = field(default_factory=SchedulerPolicy)
    dependencies: SchedulerDependencyProbe = field(default_factory=SchedulerDependencyProbe)
    recovery: SchedulerRecoveryPolicy = field(default_factory=SchedulerRecoveryPolicy)
    job_store: JsonJobStore = field(init=False)
    manifest_store: JsonManifestStore = field(init=False)
    event_store: JsonlEventStore = field(init=False)
    heartbeat: SchedulerHeartbeat = field(init=False)

    def __post_init__(self) -> None:
        self.job_store = JsonJobStore(self.jobs_dir)
        self.manifest_store = JsonManifestStore()
        self.event_store = JsonlEventStore(self.jobs_dir)
        self.heartbeat = SchedulerHeartbeat(self.jobs_dir)

    def register(self, runner: StageRunner) -> None:
        self.registry.register(runner)

    def recover_interrupted(self) -> int:
        return len(self.job_store.recover_interrupted())

    def run_once_with_lock(self, clear_lock: bool = False) -> bool:
        with SchedulerLock(self.jobs_dir, clear_lock=clear_lock):
            stalled = self.recover_stalled_jobs()
            recovered = self.recover_interrupted()
            requeued = self.auto_requeue_failed_jobs()
            if recovered or stalled or requeued:
                self.heartbeat.write(active=["recovered"])
            ran = self.run_one_ready_stage()
            self.heartbeat.write(active=[])
            self.heartbeat.clear()
            return ran

    def run_one_ready_stage(self) -> bool:
        self.recover_stalled_jobs()
        self.auto_requeue_failed_jobs()
        selection = self.select_ready_stage()
        if selection is None:
            return False
        runner = self.registry.get(selection.stage)
        if runner is None:
            return False
        self._run_stage(selection.job, selection.stage, runner)
        return True

    def _dependencies_ready(self, job: Job, stage: StageName) -> bool:
        ready, _reason = self.dependencies.ready(job, stage)
        return ready

    def select_ready_stage(self) -> SchedulerSelection | None:
        return self.policy.select_ready_job(
            self.job_store.discover(),
            self.job_store.next_stage,
            registered_stages=self.registry.registered_stages(),
            can_start=self._dependencies_ready,
        )

    def begin_stage(
        self,
        job: Job,
        stage: StageName,
        *,
        stage_run_token: str | None = None,
    ) -> tuple[Job, int, Path]:
        state = self.job_store.mark_running(job.job_id, stage, stage_run_token=stage_run_token)
        job = self.job_store.load(job.job_id)
        attempt = int(state.attempts.get(stage, 1))
        log_path = job.job_dir / "logs" / f"{stage.value}.log"
        self.manifest_store.mark_running(job.job_dir, job.task, stage, attempt, log_path)
        self.event_store.append(JobEvent(job.job_id, JobEventType.STAGE_STARTED, stage=stage, status="running", attempt=attempt))
        return job, attempt, log_path

    def attach_stage_process(
        self,
        job_id: str,
        stage: StageName,
        *,
        stage_pid: int,
        stage_run_token: str,
    ) -> bool:
        job = self.job_store.load(job_id)
        state = job.state
        if state.status != JobStatus.RUNNING or state.current_stage != stage or state.stage_run_token != stage_run_token:
            return False
        state.stage_pid = int(stage_pid)
        self.job_store.save_state(job_id, state)
        return True

    def stage_run_matches(self, job_id: str, stage: StageName, stage_run_token: str) -> bool:
        job = self.job_store.load(job_id)
        state = job.state
        return state.status == JobStatus.RUNNING and state.current_stage == stage and (state.stage_run_token or "") == stage_run_token

    def finish_stage_exception(
        self,
        job_id: str,
        stage: StageName,
        error: str,
        *,
        attempt: int,
        duration_sec: float,
        log_path: Path,
        stage_run_token: str,
    ) -> StageResult:
        if not self.stage_run_matches(job_id, stage, stage_run_token):
            return StageResult(status="skipped", skipped=True, warnings=["Stage run token no longer matches"])
        job = self.job_store.load(job_id)
        if attempt <= self.max_stage_retries:
            state = job.state
            state.status = JobStatus.PENDING
            state.current_stage = None
            state.stage_started_at = None
            state.stage_pid = None
            state.stage_run_token = None
            state.error = error
            self.job_store.save_state(job_id, state)
            self.manifest_store.mark_finished(job.job_dir, job.task, stage, "retrying", error=error, log_path=log_path)
            self.event_store.append(
                JobEvent(
                    job_id,
                    JobEventType.STAGE_RETRYING,
                    stage=stage,
                    status="retrying",
                    attempt=attempt,
                    duration_sec=duration_sec,
                    error=error,
                )
            )
            return StageResult(status="retrying", warnings=[error])
        self.job_store.mark_failed(job_id, stage, error)
        self.manifest_store.mark_finished(job.job_dir, job.task, stage, "failed", error=error, log_path=log_path)
        self.event_store.append(
            JobEvent(
                job_id,
                JobEventType.STAGE_FAILED,
                stage=stage,
                status="failed",
                attempt=attempt,
                duration_sec=duration_sec,
                error=error,
            )
        )
        return StageResult(status="failed", warnings=[error])

    def finish_stage_result(
        self,
        job_id: str,
        stage: StageName,
        result: StageResult,
        *,
        attempt: int,
        duration_sec: float,
        log_path: Path,
        stage_run_token: str,
    ) -> StageResult:
        if not self.stage_run_matches(job_id, stage, stage_run_token):
            return StageResult(status="skipped", skipped=True, warnings=["Stage run token no longer matches"])
        job = self.job_store.load(job_id)
        status = "skipped" if result.skipped else result.status
        if status == "failed":
            error = "; ".join(result.warnings) or str(result.outputs.get("error") or f"{stage.value} failed")
            self.job_store.mark_failed(job_id, stage, error)
            self.manifest_store.mark_finished(job.job_dir, job.task, stage, "failed", outputs=result.outputs, error=error, log_path=log_path)
            self.event_store.append(
                JobEvent(
                    job_id,
                    JobEventType.STAGE_FAILED,
                    stage=stage,
                    status="failed",
                    attempt=attempt,
                    duration_sec=duration_sec,
                    outputs=result.outputs,
                    error=error,
                )
            )
            return result
        state = self.job_store.mark_done(job_id, stage, result.outputs)
        self.manifest_store.mark_finished(job.job_dir, job.task, stage, status, outputs=result.outputs, log_path=log_path)
        self.event_store.append(
            JobEvent(
                job_id,
                JobEventType.STAGE_FINISHED,
                stage=stage,
                status=status,
                attempt=attempt,
                duration_sec=duration_sec,
                outputs=result.outputs,
            )
        )
        if state.status == JobStatus.DONE and bool(job.task.get("archive_on_done", True)):
            self.job_store.archive_done(job_id)
        return result

    def retry_failed(self, job_id: str) -> dict[str, str | None]:
        state = self.job_store.retry_failed(job_id)
        self.event_store.append(
            JobEvent(
                job_id,
                JobEventType.JOB_RETRY_REQUESTED,
                stage=state.failed_stage,
                status=state.status.value,
                message="Failed job moved back to pending",
            )
        )
        return {
            "job_id": job_id,
            "status": state.status.value,
            "failed_stage": state.failed_stage.value if state.failed_stage else None,
        }

    def auto_requeue_failed_jobs(self) -> int:
        requeued = 0
        for job in self.job_store.discover():
            if not self.recovery.should_auto_requeue(job):
                continue
            failed_stage = job.state.failed_stage or self.job_store.next_stage(job.state)
            if failed_stage is None:
                continue
            count = job.state.auto_requeue_count
            state = self.job_store.reset_from_stage(job.job_id, failed_stage)
            state.auto_requeue_count = count + 1
            state.error = f"Auto-requeued after failure (attempt {state.auto_requeue_count}/{self.recovery.max_auto_requeues})"
            self.job_store.save_state(job.job_id, state)
            self.event_store.append(
                JobEvent(
                    job.job_id,
                    JobEventType.JOB_RETRY_REQUESTED,
                    stage=failed_stage,
                    status=state.status.value,
                    message=state.error,
                )
            )
            requeued += 1
        return requeued

    def recover_stalled_jobs(self) -> int:
        recovered = 0
        for job in self.job_store.discover():
            if not self.recovery.is_stalled(job):
                continue
            stage = job.state.current_stage
            if stage is None:
                continue
            attempt = int(job.state.attempts.get(stage, 1))
            error = f"Stage {stage.value} stalled after {self.recovery.timeout_for(stage)}s without progress"
            if attempt <= self.max_stage_retries:
                state = job.state
                state.status = JobStatus.PENDING
                state.current_stage = None
                state.failed_stage = stage
                state.stage_started_at = None
                state.stage_pid = None
                state.stage_run_token = None
                state.error = error
                self.job_store.save_state(job.job_id, state)
                self.manifest_store.mark_finished(job.job_dir, job.task, stage, "retrying", error=error)
                self.event_store.append(
                    JobEvent(
                        job.job_id,
                        JobEventType.STAGE_RETRYING,
                        stage=stage,
                        status="retrying",
                        attempt=attempt,
                        error=error,
                    )
                )
            else:
                state = self.job_store.mark_failed(job.job_id, stage, error)
                state.stage_pid = None
                state.stage_run_token = None
                self.job_store.save_state(job.job_id, state)
                self.manifest_store.mark_finished(job.job_dir, job.task, stage, "failed", error=error)
                self.event_store.append(
                    JobEvent(
                        job.job_id,
                        JobEventType.STAGE_FAILED,
                        stage=stage,
                        status="failed",
                        attempt=attempt,
                        error=error,
                    )
                )
            recovered += 1
        return recovered

    def reset_from_stage(self, job_id: str, stage: StageName | str) -> dict[str, object]:
        reset_stage = StageName(str(stage))
        state = self.job_store.reset_from_stage(job_id, reset_stage)
        self.event_store.append(
            JobEvent(
                job_id,
                JobEventType.JOB_RESET_REQUESTED,
                stage=reset_stage,
                status=state.status.value,
                message=f"Job reset from stage: {reset_stage.value}",
            )
        )
        return {
            "job_id": job_id,
            "status": state.status.value,
            "reset_stage": reset_stage.value,
            "completed_stages": [item.value for item in state.completed_stages],
        }

    def _run_stage(self, job: Job, stage: StageName, runner: StageRunner) -> StageResult:
        job, attempt, log_path = self.begin_stage(job, stage)
        started_at = monotonic()

        try:
            result = runner.run(
                StageContext(
                    job_id=job.job_id,
                    job_dir=job.job_dir,
                    task=job.task,
                    stage=stage,
                    attempt=attempt,
                    artifacts=job.state.artifacts,
                )
            )
        except Exception as exc:
            return self.finish_stage_exception(
                job.job_id,
                stage,
                str(exc),
                attempt=attempt,
                duration_sec=monotonic() - started_at,
                log_path=log_path,
                stage_run_token=job.state.stage_run_token or "",
            )
        return self.finish_stage_result(
            job.job_id,
            stage,
            result,
            attempt=attempt,
            duration_sec=monotonic() - started_at,
            log_path=log_path,
            stage_run_token=job.state.stage_run_token or "",
        )
