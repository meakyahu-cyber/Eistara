from __future__ import annotations

import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from eistara.core.jobs import Job, StageName

from .lock import SchedulerLock
from .service import SchedulerService
from .worker_result import read_stage_worker_result


@dataclass(slots=True)
class ActiveStageProcess:
    token: str
    job_id: str
    stage: StageName
    attempt: int
    process: subprocess.Popen
    result_path: Path
    log_path: Path
    log_file: TextIO
    started_at: float

    @property
    def label(self) -> str:
        return f"{self.job_id}:{self.stage.value}:{self.process.pid}"


@dataclass(frozen=True, slots=True)
class SchedulerProcessTick:
    launched: int = 0
    finished: int = 0
    recovered: int = 0
    requeued: int = 0
    launch_failures: int = 0

    @property
    def did_work(self) -> bool:
        return bool(self.launched or self.finished or self.recovered or self.requeued or self.launch_failures)

    def merge(self, other: "SchedulerProcessTick") -> "SchedulerProcessTick":
        return SchedulerProcessTick(
            launched=self.launched + other.launched,
            finished=self.finished + other.finished,
            recovered=self.recovered + other.recovered,
            requeued=self.requeued + other.requeued,
            launch_failures=self.launch_failures + other.launch_failures,
        )


@dataclass(slots=True)
class SchedulerProcessSupervisor:
    service: SchedulerService
    preset: str
    config_path: str | Path | None = None
    render_audio: bool = False
    render_video: bool = False
    python_executable: str = sys.executable
    worker_module: str = "eistara.runtime.worker"
    cwd: str | Path | None = None
    active: dict[str, ActiveStageProcess] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.config_path:
            self.config_path = Path(self.config_path).expanduser().resolve()
        if self.cwd is None:
            self.cwd = Path(__file__).resolve().parents[3]
        else:
            self.cwd = Path(self.cwd).expanduser().resolve()

    def run_once(self, *, launch: bool = True) -> SchedulerProcessTick:
        finished = self.reap_finished()
        recovered = self.recover_stalled_jobs()
        requeued = self.service.auto_requeue_failed_jobs()
        launched = 0
        launch_failures = 0
        if launch:
            launched, launch_failures = self.launch_available()
        self._sync_heartbeat()
        return SchedulerProcessTick(
            launched=launched,
            finished=finished,
            recovered=recovered,
            requeued=requeued,
            launch_failures=launch_failures,
        )

    def run_once_with_lock(self, *, clear_lock: bool = False, wait: bool = False, poll_interval: float = 0.25) -> SchedulerProcessTick:
        with SchedulerLock(self.service.jobs_dir, clear_lock=clear_lock):
            tick = self.run_once(launch=True)
            if wait:
                tick = tick.merge(self.wait_for_active(poll_interval=poll_interval))
            self._sync_heartbeat()
            return tick

    def wait_for_active(self, *, poll_interval: float = 0.25, timeout_sec: float | None = None) -> SchedulerProcessTick:
        total = SchedulerProcessTick()
        deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
        while self.active:
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(max(0.01, float(poll_interval)))
            total = total.merge(self.run_once(launch=False))
        return total

    def launch_available(self) -> tuple[int, int]:
        launched = 0
        failures = 0
        while True:
            selection = self.service.select_ready_stage()
            if selection is None:
                return launched, failures
            active = self.launch_stage(selection.job, selection.stage)
            if active is None:
                failures += 1
                return launched, failures
            launched += 1

    def launch_stage(self, job: Job, stage: StageName) -> ActiveStageProcess | None:
        token = uuid.uuid4().hex
        started_at = time.monotonic()
        job, attempt, log_path = self.service.begin_stage(job, stage, stage_run_token=token)
        result_path = job.job_dir / "logs" / f"{stage.value}.{token}.result.json"
        command = self._worker_command(job.job_id, stage, token, result_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("a", encoding="utf-8", errors="replace")
        log_file.write(f"\n--- eistara-worker {stage.value} token={token} ---\n")
        log_file.write(" ".join(command) + "\n")
        log_file.flush()
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            env["PYTHONUNBUFFERED"] = "1"
            env["EISTARA_JOB_DIR"] = os.fspath(job.job_dir)
            env["EISTARA_OUTPUT_DIR"] = os.fspath(Path(job.task.get("output_dir") or job.job_dir / "output"))
            process = subprocess.Popen(
                command,
                cwd=os.fspath(self.cwd) if self.cwd else None,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            log_file.write(f"Failed to launch worker: {exc}\n")
            log_file.close()
            self.service.finish_stage_exception(
                job.job_id,
                stage,
                str(exc),
                attempt=attempt,
                duration_sec=time.monotonic() - started_at,
                log_path=log_path,
                stage_run_token=token,
            )
            return None

        active = ActiveStageProcess(
            token=token,
            job_id=job.job_id,
            stage=stage,
            attempt=attempt,
            process=process,
            result_path=result_path,
            log_path=log_path,
            log_file=log_file,
            started_at=started_at,
        )
        if not self.service.attach_stage_process(job.job_id, stage, stage_pid=process.pid, stage_run_token=token):
            self._terminate_active(active)
            self.service.finish_stage_exception(
                job.job_id,
                stage,
                "Worker launched but stage state changed before PID could be attached",
                attempt=attempt,
                duration_sec=time.monotonic() - started_at,
                log_path=log_path,
                stage_run_token=token,
            )
            return None
        self.active[token] = active
        return active

    def reap_finished(self) -> int:
        finished = 0
        for token, active in list(self.active.items()):
            returncode = active.process.poll()
            if returncode is None:
                continue
            active.log_file.close()
            self.active.pop(token, None)
            duration = time.monotonic() - active.started_at
            if not self.service.stage_run_matches(active.job_id, active.stage, active.token):
                finished += 1
                continue
            try:
                worker_result = read_stage_worker_result(active.result_path)
            except Exception as exc:
                self.service.finish_stage_exception(
                    active.job_id,
                    active.stage,
                    f"Worker exited with code {returncode} and no readable result: {exc}",
                    attempt=active.attempt,
                    duration_sec=duration,
                    log_path=active.log_path,
                    stage_run_token=active.token,
                )
                finished += 1
                continue

            if returncode != 0 or worker_result.status == "exception":
                error = worker_result.error or f"Worker exited with code {returncode}"
                self.service.finish_stage_exception(
                    active.job_id,
                    active.stage,
                    error,
                    attempt=active.attempt,
                    duration_sec=duration,
                    log_path=active.log_path,
                    stage_run_token=active.token,
                )
            else:
                self.service.finish_stage_result(
                    active.job_id,
                    active.stage,
                    worker_result.to_stage_result(),
                    attempt=active.attempt,
                    duration_sec=duration,
                    log_path=active.log_path,
                    stage_run_token=active.token,
                )
            finished += 1
        return finished

    def recover_stalled_jobs(self) -> int:
        recovered = self.service.recover_stalled_jobs()
        for token, active in list(self.active.items()):
            if self.service.stage_run_matches(active.job_id, active.stage, active.token):
                continue
            self._terminate_active(active)
            self.active.pop(token, None)
        return recovered

    def terminate_all(self) -> None:
        for token, active in list(self.active.items()):
            self._terminate_active(active)
            self.active.pop(token, None)
        self._sync_heartbeat()

    def active_labels(self) -> list[str]:
        return [active.label for active in self.active.values()]

    def _worker_command(self, job_id: str, stage: StageName, token: str, result_path: Path) -> list[str]:
        command = [
            self.python_executable,
            "-m",
            self.worker_module,
            "--jobs-dir",
            os.fspath(self.service.jobs_dir),
            "--job-id",
            job_id,
            "--stage",
            stage.value,
            "--preset",
            self.preset,
            "--result-json",
            os.fspath(result_path),
            "--run-token",
            token,
        ]
        if self.config_path:
            command.extend(["--config", os.fspath(self.config_path)])
        if self.render_audio:
            command.append("--render-audio")
        if self.render_video:
            command.append("--render-video")
        return command

    def _terminate_active(self, active: ActiveStageProcess) -> None:
        try:
            if active.process.poll() is None:
                active.process.terminate()
                try:
                    active.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    active.process.kill()
                    active.process.wait(timeout=5)
        finally:
            try:
                active.log_file.close()
            except OSError:
                pass

    def _sync_heartbeat(self) -> None:
        if self.active:
            self.service.heartbeat.write(active=self.active_labels())
        else:
            self.service.heartbeat.clear()
