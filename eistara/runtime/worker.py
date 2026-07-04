from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

from eistara.config import ConfigLoader
from eistara.core.jobs import JobStatus, JsonJobStore, StageName
from eistara.core.pipeline import StageContext, resolve_task_output_dir
from eistara.core.scheduler.worker_result import StageWorkerResult, write_stage_worker_result

from .pipeline import build_runners


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def run_stage_worker(
    *,
    jobs_dir: str | Path,
    job_id: str,
    stage: str | StageName,
    preset: str,
    result_path: str | Path,
    config_path: str | Path | None = None,
    render_audio: bool = False,
    render_video: bool = False,
    run_token: str | None = None,
) -> int:
    stage_name = StageName(str(stage))
    result_file = Path(result_path)
    try:
        config = ConfigLoader(config_path).load() if config_path else ConfigLoader().load()
        runner = next((item for item in build_runners(preset, config=config, render_audio=render_audio, render_video=render_video) if item.stage == stage_name), None)
        if runner is None:
            raise ValueError(f"No runner registered for stage: {stage_name.value}")

        job = JsonJobStore(jobs_dir).load(job_id)
        if job.state.status != JobStatus.RUNNING or job.state.current_stage != stage_name:
            raise RuntimeError(f"Job is not running requested stage: {job_id}:{stage_name.value}")
        if run_token and job.state.stage_run_token != run_token:
            raise RuntimeError(f"Stage run token mismatch for {job_id}:{stage_name.value}")

        attempt = int(job.state.attempts.get(stage_name, 1))
        previous_job_dir = os.environ.get("EISTARA_JOB_DIR")
        previous_output_dir = os.environ.get("EISTARA_OUTPUT_DIR")
        os.environ["EISTARA_JOB_DIR"] = os.fspath(job.job_dir)
        os.environ["EISTARA_OUTPUT_DIR"] = os.fspath(resolve_task_output_dir(job.job_dir, job.task))
        try:
            result = runner.run(
                StageContext(
                    job_id=job.job_id,
                    job_dir=job.job_dir,
                    task=job.task,
                    stage=stage_name,
                    attempt=attempt,
                    artifacts=job.state.artifacts,
                )
            )
        finally:
            _restore_env("EISTARA_JOB_DIR", previous_job_dir)
            _restore_env("EISTARA_OUTPUT_DIR", previous_output_dir)
        write_stage_worker_result(result_file, StageWorkerResult.from_stage_result(job.job_id, stage_name, result))
        return 0
    except Exception as exc:
        traceback.print_exc()
        write_stage_worker_result(
            result_file,
            StageWorkerResult.from_exception(job_id, stage_name, str(exc), traceback.format_exc()),
        )
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eistara-stage-worker")
    parser.add_argument("--jobs-dir", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stage", required=True, choices=[stage.value for stage in StageName])
    parser.add_argument("--preset", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--config")
    parser.add_argument("--run-token")
    parser.add_argument("--render-audio", action="store_true")
    parser.add_argument("--render-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
    args = build_parser().parse_args(argv)
    return run_stage_worker(
        jobs_dir=args.jobs_dir,
        job_id=args.job_id,
        stage=args.stage,
        preset=args.preset,
        result_path=args.result_json,
        config_path=args.config,
        render_audio=args.render_audio,
        render_video=args.render_video,
        run_token=args.run_token,
    )


if __name__ == "__main__":
    raise SystemExit(main())
