from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from eistara.config import ConfigLoader
from eistara.config.loader import DEFAULT_LOCAL_CONFIG, deep_merge, load_mapping
from eistara.config.youtube_cookies import apply_youtube_cookie_config, browser_cookie_candidates, write_mapping
from eistara.adapters.asr.audio_separator import audio_separator_model_status
from eistara.adapters.llm import OpenAICompatibleLlmClient, OpenAICompatibleSettings, RequestsHttpTransport
from eistara.core.jobs import Job, JobFactory, JsonJobStore, JobStatus, STAGE_ORDER, StageName, history_dir_for_jobs
from eistara.core.jobs.factory import normalize_task
from eistara.core.observability import JsonlEventStore
from eistara.core.scheduler import SchedulerLock, collect_status_rows, recover_orphaned_scheduler_state, scheduler_health
from eistara.runtime import WEBUI_DEFAULT_PRESET, RuntimeHealthService, build_model_dependency_report, build_scheduler

from apps.webui.diagnostics import build_diagnostic_package, build_diagnostic_summary, render_diagnostic_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_FILE = ".webui_scheduler.json"


@dataclass(frozen=True, slots=True)
class WebUiSettings:
    jobs_dir: Path
    config_path: Path | None = None
    preset: str = WEBUI_DEFAULT_PRESET


class WebUiBackend:
    def __init__(self, settings: WebUiSettings):
        self.settings = settings
        self._health_cache: dict[bool, dict[str, Any]] = {}

    @property
    def jobs_dir(self) -> Path:
        return _resolve_project_path(self.settings.jobs_dir)

    def dashboard(self, *, include_history: bool = True) -> dict[str, Any]:
        rows = collect_status_rows(self.jobs_dir, include_history=include_history)
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {"counts": counts, "jobs": rows}

    def active_dashboard(self) -> dict[str, Any]:
        return self.dashboard(include_history=False)

    def history_dashboard(self) -> dict[str, Any]:
        rows = [row for row in self.dashboard(include_history=True)["jobs"] if row.get("location") == "history"]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        return {"counts": counts, "jobs": rows}

    def config_dict(self) -> dict[str, Any]:
        return ConfigLoader(self.settings.config_path).load_dict() if self.settings.config_path else ConfigLoader().load_dict()

    def update_config(self, updates: dict[str, Any]) -> dict[str, Any]:
        config_path = self._config_write_path()
        current = load_mapping(config_path)
        patch: dict[str, Any] = {}
        for key, value in updates.items():
            _nested_set(patch, tuple(str(key).split(".")), value)
        updated = deep_merge(current, patch)
        write_mapping(config_path, updated)
        return {"updated": True, "config_path": str(config_path), "keys": sorted(updates)}

    def active_config(self) -> dict[str, Any]:
        config_path = self._config_write_path()
        config = self._config()
        return {
            "path": str(config_path),
            "exists": config_path.exists(),
            "api": {
                "base_url": config.api.base_url,
                "model": config.api.model,
                "has_key": bool(config.api.key),
                "llm_support_json": config.api.llm_support_json,
                "proxy_url": config.api.proxy_url,
                "trust_env_proxy": config.api.trust_env_proxy,
            },
            "vocal_separation": self._vocal_separation_summary(config),
        }

    def vocal_separation_status(self) -> dict[str, Any]:
        return self._vocal_separation_summary(self._config())

    def model_dependency_report(self) -> dict[str, Any]:
        return build_model_dependency_report(self._config(), project_root=PROJECT_ROOT).to_dict()

    def health(self, *, probe_llm: bool = False) -> dict[str, Any]:
        if probe_llm in self._health_cache:
            return self._health_cache[probe_llm]
        config = self._config()
        tts_api_url = str(config.indextts.get("api_url") or "") if config.tts_method == "indextts" else None
        runtime = RuntimeHealthService().check(
            llm_base_url=config.api.base_url if probe_llm else None,
            llm_api_key=config.api.key if probe_llm else None,
            llm_model=config.api.model if probe_llm else None,
            llm_support_json=config.api.llm_support_json if probe_llm else False,
            llm_proxy_url=config.api.proxy_url if probe_llm else None,
            llm_trust_env_proxy=config.api.trust_env_proxy if probe_llm else True,
            tts_api_url=tts_api_url,
            tts_label=f"TTS ({config.tts_method})",
        ).to_dict()
        tts_checks = [check for check in runtime["checks"] if check.get("kind") == "tts"]
        health = {
            "scheduler": scheduler_health(self.jobs_dir),
            "runtime": runtime,
            "preset": self.settings.preset,
            "tts": {
                "method": config.tts_method,
                "api_url": tts_api_url or "",
                "checks": tts_checks,
                "ok": all(check.get("ok") for check in tts_checks) if tts_checks else None,
            },
            "model_dependencies": build_model_dependency_report(config, project_root=PROJECT_ROOT).to_dict(),
            "youtube_cookies": self.youtube_cookies(),
        }
        self._health_cache[probe_llm] = health
        return health

    def youtube_cookies(self) -> dict[str, Any]:
        config = self._config()
        return {
            "configured_browser": config.youtube.cookies_from_browser,
            "configured_profile": config.youtube.cookies_browser_profile,
            "candidates": [candidate.to_dict() for candidate in browser_cookie_candidates()],
        }

    def configure_youtube_cookies(self, *, browser: str = "auto", profile: str = "", dry_run: bool = False) -> dict[str, Any]:
        return apply_youtube_cookie_config(self._config_write_path(), browser=browser, profile=profile, dry_run=dry_run)

    def list_llm_models(self) -> dict[str, Any]:
        config = self._config()
        client = OpenAICompatibleLlmClient(
            OpenAICompatibleSettings(
                base_url=config.api.base_url,
                model=config.api.model,
                api_key=config.api.key,
                timeout_sec=min(float(config.api.timeout_sec), 30.0),
                user_agent=config.api.user_agent,
                trust_env_proxy=config.api.trust_env_proxy,
                proxy_url=config.api.proxy_url,
                max_retries=0,
                retry_base_delay_sec=config.api.retry_base_delay_sec,
                retry_max_delay_sec=config.api.retry_max_delay_sec,
            ),
            transport=RequestsHttpTransport(trust_env=config.api.trust_env_proxy, proxy_url=config.api.proxy_url),
        )
        response = client.list_models()
        models = _model_ids_from_response(response)
        return {"models": models, "count": len(models)}

    def check_llm_api(self) -> dict[str, Any]:
        config = self._config()
        client = OpenAICompatibleLlmClient(
            OpenAICompatibleSettings(
                base_url=config.api.base_url,
                model=config.api.model,
                api_key=config.api.key,
                timeout_sec=min(float(config.api.timeout_sec), 30.0),
                response_format_json=config.api.llm_support_json,
                user_agent=config.api.user_agent,
                trust_env_proxy=config.api.trust_env_proxy,
                proxy_url=config.api.proxy_url,
                max_retries=0,
                outer_retries=0,
                retry_base_delay_sec=config.api.retry_base_delay_sec,
                retry_max_delay_sec=config.api.retry_max_delay_sec,
                persist_cache=False,
            ),
            transport=RequestsHttpTransport(trust_env=config.api.trust_env_proxy, proxy_url=config.api.proxy_url),
        )
        response = client.ask_json(
            "This is a test, response 'message':'success' in json format.",
            log_title="webui_api_check",
            use_cache=False,
        )
        ok = isinstance(response, dict) and str(response.get("message") or "").lower() == "success"
        return {"ok": ok, "response": response}

    def create_single_job(
        self,
        source: str,
        *,
        resolution: str = "",
        source_language: str = "",
        target_language: str = "",
        clear_existing: bool = False,
    ) -> dict[str, Any]:
        if clear_existing:
            self.clear_jobs()
        task = {
            "source": source,
            "title": "Single Video",
            "resolution": resolution,
            "source_language": source_language,
            "target_language": target_language,
            "archive_on_done": False,
        }
        created = JobFactory(
            self.jobs_dir,
            project_root=PROJECT_ROOT,
            config_path=self._config_write_path(),
        ).create_from_tasks([normalize_task(task, 0, project_root=PROJECT_ROOT)])
        job_id = created[0].name if created else ""
        return {"created": len(created), "job": job_id}

    def save_upload(self, filename: str, data: bytes) -> Path:
        upload_dir = self.jobs_dir / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _safe_filename(filename)
        path = upload_dir / safe_name
        path.write_bytes(data)
        return path

    def save_upload_source(self, filename: str, data: bytes) -> Path:
        path = self.save_upload(filename, data)
        config = self.config_dict()
        audio_formats = _format_set(config.get("allowed_audio_formats") or ("wav", "mp3", "flac", "m4a"))
        if path.suffix.lower().lstrip(".") not in audio_formats:
            return path
        return self._convert_audio_upload_to_video(path)

    def _convert_audio_upload_to_video(self, audio_path: Path) -> Path:
        output_video = audio_path.with_name(f"{audio_path.stem}_black_screen.mp4")
        if output_video.exists():
            return output_video
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=640x360",
            "-i",
            os.fspath(audio_path),
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            os.fspath(output_video),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
        try:
            audio_path.unlink()
        except OSError:
            pass
        return output_video

    def clear_jobs(self) -> dict[str, Any]:
        root = self.jobs_dir
        _validate_cleanup_root(root)
        root.mkdir(parents=True, exist_ok=True)
        removed = 0
        for child in root.iterdir():
            if not child.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Refusing to remove path outside jobs dir: {child}")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed += 1
        return {"removed": removed, "jobs_dir": str(root)}

    def delete_active_job(self, job_id: str) -> dict[str, Any]:
        root = self.jobs_dir.resolve()
        target = (root / job_id).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Refusing to remove path outside jobs dir: {target}")
        if target == root or not (target / "task.json").exists():
            raise FileNotFoundError(f"Active job not found: {job_id}")
        shutil.rmtree(target)
        return {"deleted": True, "job": job_id, "path": str(target)}

    def create_jobs_from_sources(
        self,
        sources_text: str,
        *,
        title_prefix: str = "",
        resolution: str = "",
        source_language: str = "",
        target_language: str = "",
    ) -> dict[str, Any]:
        rows = []
        for index, line in enumerate(sources_text.splitlines(), start=1):
            source = line.strip()
            if not source or source.startswith("#"):
                continue
            title = f"{title_prefix}{index}" if title_prefix else ""
            rows.append(
                {
                    "source": source,
                    "title": title,
                    "resolution": resolution,
                    "source_language": source_language,
                    "target_language": target_language,
                    "archive_on_done": True,
                }
            )
        tasks = [
            normalize_task(row, index, project_root=PROJECT_ROOT)
            for index, row in enumerate(rows)
        ]
        created = JobFactory(
            self.jobs_dir,
            project_root=PROJECT_ROOT,
            config_path=self._config_write_path(),
        ).create_from_tasks(tasks)
        return {"created": len(created), "jobs": [path.name for path in created]}

    def latest_job_id(self, *, include_history: bool = True) -> str | None:
        jobs = [*JsonJobStore(self.jobs_dir).discover()]
        if include_history:
            jobs.extend(JsonJobStore(history_dir_for_jobs(self.jobs_dir)).discover())
        jobs = sorted(jobs, key=lambda job: job.state.updated_at)
        return jobs[-1].job_id if jobs else None

    def latest_active_job_id(self) -> str | None:
        return self.latest_job_id(include_history=False)

    def archive_job(self, job_id: str) -> dict[str, Any]:
        target = JsonJobStore(self.jobs_dir).archive_done(job_id)
        if target is None:
            raise ValueError(f"Job is not completed or cannot be archived: {job_id}")
        return {"archived": True, "job": target.name, "path": str(target)}

    def job_detail(
        self,
        job_id: str,
        *,
        include_events: bool = True,
        include_reports: bool = True,
    ) -> dict[str, Any]:
        job = self._load_job(job_id)
        return {
            "job_id": job.job_id,
            "job_dir": str(job.job_dir),
            "task": job.task,
            "state": job.state.to_dict(),
            "outputs": self._job_outputs_for_job(job),
            "manifest": self._read_json(job.job_dir / "manifest.json") if include_reports else None,
            "quality_report": self._read_json(job.job_dir / "output" / "quality_report.json") if include_reports else None,
            "events": [event.to_dict() for event in self._read_job_events(job)] if include_events else [],
        }

    def job_outputs(self, job_id: str) -> list[dict[str, Any]]:
        job = self._load_job(job_id)
        return self._job_outputs_for_job(job)

    def job_diagnostic_summary(self, job_id: str) -> dict[str, Any]:
        job = self._load_job(job_id)
        events = [event.to_dict() for event in self._read_job_events(job)]
        return build_diagnostic_summary(
            job=job,
            outputs=self._job_outputs_for_job(job),
            manifest=self._read_json(job.job_dir / "manifest.json"),
            quality_report=self._read_json(job.job_dir / "output" / "quality_report.json"),
            events=events,
            config_summary=self.active_config(),
        )

    def job_diagnostic_text(self, job_id: str) -> str:
        return render_diagnostic_text(self.job_diagnostic_summary(job_id))

    def build_job_diagnostic_package(self, job_id: str) -> dict[str, Any]:
        job = self._load_job(job_id)
        config_path = self._config_write_path()
        package_path = build_diagnostic_package(
            job=job,
            summary=self.job_diagnostic_summary(job_id),
            config=load_mapping(config_path) if config_path.exists() else {},
            config_source=config_path,
            scheduler_log=self.jobs_dir / "scheduler.webui.log",
        )
        return {
            "path": str(package_path),
            "filename": package_path.name,
            "size": package_path.stat().st_size if package_path.exists() else 0,
        }

    def _job_outputs_for_job(self, job: Job) -> list[dict[str, Any]]:
        output_dir = Path(job.task.get("output_dir") or job.job_dir / "output")
        artifacts = job.state.artifacts
        candidates = [
            ("source_video", artifacts.get("source_video"), "video"),
            ("source_subtitle", artifacts.get("source_srt"), "subtitle"),
            ("translated_subtitle", artifacts.get("translated_srt"), "subtitle"),
            ("dub_subtitle", artifacts.get("dub_subtitles"), "subtitle"),
            ("dub_bilingual_subtitle", artifacts.get("dub_bilingual_subtitles"), "subtitle"),
            ("dub_audio", artifacts.get("dub_audio"), "audio"),
            ("dub_video", artifacts.get("dub_video"), "video"),
            ("translations", artifacts.get("translations_json"), "internal"),
            ("subtitle_rows", artifacts.get("subtitle_rows_json"), "internal"),
        ]
        fallback_names = {
            "source_video": ("source_video.mp4", "source_video.webm"),
            "source_subtitle": ("src.srt",),
            "translated_subtitle": ("trans.srt",),
            "dub_subtitle": ("output_dub.srt",),
            "dub_bilingual_subtitle": ("output_dub_trans_src.srt",),
            "dub_audio": ("dub.mp3",),
            "dub_video": ("output_dub.mp4",),
            "translations": ("internal/translations.json", "translations.json"),
            "subtitle_rows": ("internal/subtitle_rows.json", "subtitle_rows.json"),
        }
        rows: list[dict[str, Any]] = []
        for role, configured, kind in candidates:
            path = _first_existing_path(configured, *(output_dir / name for name in fallback_names.get(role, ())))
            if path is None:
                continue
            rows.append(
                {
                    "role": role,
                    "kind": kind,
                    "path": str(path),
                    "filename": path.name,
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.exists() and path.is_file() else 0,
                }
            )
        return rows

    def recent_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        events = JsonlEventStore(self.jobs_dir).read_all()
        events.extend(JsonlEventStore(history_dir_for_jobs(self.jobs_dir)).read_all())
        events = sorted(events, key=lambda event: event.created_at)
        return [event.to_dict() for event in events[-max(0, int(limit)) :]]

    def scheduler_safety(self) -> dict[str, Any]:
        return scheduler_health(self.jobs_dir)

    def run_once(self, *, preset: str | None = None, clear_lock: bool = True) -> dict[str, Any]:
        config = self._config()
        service = build_scheduler(
            self.jobs_dir,
            preset=preset or self.settings.preset,
            config=config,
            max_stage_retries=config.batch.max_stage_retries,
        )
        ran = service.run_once_with_lock(clear_lock=clear_lock)
        return {"ran": ran, "preset": preset or self.settings.preset}

    def run_until_stage(self, stage: str, *, preset: str | None = None, max_steps: int = 100) -> dict[str, Any]:
        target_stage = StageName(stage)
        config = self._config()
        service = build_scheduler(
            self.jobs_dir,
            preset=preset or self.settings.preset,
            config=config,
            max_stage_retries=config.batch.max_stage_retries,
        )
        ran_count = 0
        with SchedulerLock(self.jobs_dir, clear_lock=True):
            while ran_count < max_steps:
                jobs = service.job_store.discover()
                if not jobs:
                    return {"status": "empty", "ran_count": ran_count, "target_stage": target_stage.value}
                failed = [job for job in jobs if job.state.status == JobStatus.FAILED]
                if failed:
                    return {
                        "status": "failed",
                        "ran_count": ran_count,
                        "target_stage": target_stage.value,
                        "failed": [job.job_id for job in failed],
                    }
                if _all_jobs_reached(jobs, target_stage):
                    return {"status": "reached", "ran_count": ran_count, "target_stage": target_stage.value}
                service.recover_stalled_jobs()
                service.auto_requeue_failed_jobs()
                ran = service.run_one_ready_stage()
                if not ran:
                    return {"status": "idle", "ran_count": ran_count, "target_stage": target_stage.value}
                ran_count += 1
        return {"status": "max_steps", "ran_count": ran_count, "target_stage": target_stage.value}

    def retry_failed(self, job_id: str, *, preset: str | None = None) -> dict[str, Any]:
        config = self._config()
        service = build_scheduler(
            self.jobs_dir,
            preset=preset or self.settings.preset,
            config=config,
            max_stage_retries=config.batch.max_stage_retries,
        )
        return service.retry_failed(job_id)

    def reset_from_stage(self, job_id: str, stage: str, *, preset: str | None = None) -> dict[str, Any]:
        config = self._config()
        service = build_scheduler(
            self.jobs_dir,
            preset=preset or self.settings.preset,
            config=config,
            max_stage_retries=config.batch.max_stage_retries,
        )
        return service.reset_from_stage(job_id, StageName(stage))

    def recover(self) -> dict[str, Any]:
        count = recover_orphaned_scheduler_state(self.jobs_dir)
        return {"recovered": count}

    def reset_failed(self, *, preset: str | None = None) -> dict[str, Any]:
        config = self._config()
        service = build_scheduler(
            self.jobs_dir,
            preset=preset or self.settings.preset,
            config=config,
            max_stage_retries=config.batch.max_stage_retries,
        )
        reset_jobs: list[str] = []
        for job in JsonJobStore(self.jobs_dir).discover():
            if job.state.status != JobStatus.FAILED:
                continue
            reset_stage = job.state.failed_stage or service.job_store.next_stage(job.state)
            if reset_stage is None:
                continue
            service.reset_from_stage(job.job_id, reset_stage)
            reset_jobs.append(job.job_id)
        return {"reset": len(reset_jobs), "jobs": reset_jobs}

    def active_scheduler_pid(self) -> int | None:
        control = self._read_control()
        pid = _optional_int(control.get("pid"))
        if _pid_running(pid):
            return pid
        health = scheduler_health(self.jobs_dir)
        lock_pid = _optional_int(health.get("lock_pid"))
        if _pid_running(lock_pid):
            return lock_pid
        return None

    def start_scheduler(self, *, preset: str | None = None) -> dict[str, Any]:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        if self.active_scheduler_pid():
            return {"started": False, "pid": self.active_scheduler_pid(), "reason": "already_running"}
        if not JsonJobStore(self.jobs_dir).discover():
            return {"started": False, "pid": None, "reason": "no_jobs"}

        config = self._config()
        selected_preset = preset or self.settings.preset
        log_path = self.jobs_dir / "scheduler.webui.log"
        command = [
            sys.executable,
            "-m",
            "apps.cli.main",
            "--jobs-dir",
            os.fspath(self.jobs_dir),
        ]
        config_path = self._config_write_path()
        if config_path.exists():
            command.extend(["--config", os.fspath(config_path)])
        command.extend(
            [
                "run-loop",
                "--preset",
                selected_preset,
                "--clear-lock",
                "--poll-interval",
                str(config.batch.poll_interval_sec),
                "--stop-when-idle",
            ]
        )
        env = os.environ.copy()
        env.update({"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"})
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("ab")
        log_file.write(f"\n--- webui scheduler start {datetime.now().isoformat(timespec='seconds')} ---\n".encode("utf-8"))
        log_file.write(("Command: " + " ".join(command) + "\n").encode("utf-8", errors="replace"))
        log_file.flush()
        process = subprocess.Popen(
            command,
            cwd=os.fspath(PROJECT_ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            creationflags=creationflags,
        )
        log_file.close()
        self._write_control({"pid": process.pid, "started_at": datetime.now().isoformat(timespec="seconds"), "log": str(log_path)})
        return {"started": True, "pid": process.pid, "log": str(log_path), "preset": selected_preset}

    def stop_scheduler(self) -> dict[str, Any]:
        pid = self.active_scheduler_pid()
        stopped = False
        if pid:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, text=True, check=False)
            else:
                os.kill(pid, signal.SIGTERM)
            stopped = True
        recovered = recover_orphaned_scheduler_state(self.jobs_dir)
        self._remove_control()
        return {"stopped": stopped, "recovered": recovered}

    def _config(self):
        return ConfigLoader(self.settings.config_path).load() if self.settings.config_path else ConfigLoader().load()

    def _config_write_path(self) -> Path:
        return Path(self.settings.config_path).expanduser().resolve() if self.settings.config_path else DEFAULT_LOCAL_CONFIG

    def _vocal_separation_summary(self, config) -> dict[str, Any]:
        model_dir = _resolve_project_path(config.demucs.audio_separator_model_dir)
        model = audio_separator_model_status(model_filename=config.demucs.audio_separator_model, model_dir=model_dir)
        providers = _onnxruntime_providers()
        return {
            "enabled": config.demucs.enabled,
            "provider": config.demucs.provider,
            "segment_minutes": config.demucs.segment_minutes,
            "audio_separator_model": config.demucs.audio_separator_model,
            "audio_separator_model_dir": str(model_dir),
            "audio_separator_model_exists": model["exists"],
            "audio_separator_model_size": model["size"],
            "audio_separator_model_valid": model["valid"],
            "audio_separator_model_expected_size": model["expected_size"],
            "audio_separator_model_expected_md5": model["expected_md5"],
            "audio_separator_model_md5": model["md5"],
            "audio_separator_model_url": model["url"],
            "audio_separator_model_mirror_url": model["mirror_url"],
            "onnxruntime_providers": providers,
            "onnx_cuda": "CUDAExecutionProvider" in providers,
        }

    def _control_path(self) -> Path:
        return self.jobs_dir / CONTROL_FILE

    def _read_control(self) -> dict[str, Any]:
        path = self._control_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_control(self, data: dict[str, Any]) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._control_path().write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _remove_control(self) -> None:
        try:
            self._control_path().unlink()
        except FileNotFoundError:
            pass

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else {"value": data}

    def _load_job(self, job_id: str) -> Job:
        active_path = self.jobs_dir / job_id
        if (active_path / "task.json").exists():
            return JsonJobStore(self.jobs_dir).load(job_id)
        history_dir = history_dir_for_jobs(self.jobs_dir)
        archived_path = history_dir / job_id
        if (archived_path / "task.json").exists():
            return JsonJobStore(history_dir).load(job_id)
        raise FileNotFoundError(f"Job not found: {job_id}")

    def _read_job_events(self, job: Job):
        return JsonlEventStore(job.job_dir.parent).read_job(job.job_id)


def _resolve_project_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def _onnxruntime_providers() -> list[str]:
    try:
        import onnxruntime as ort

        return [str(provider) for provider in ort.get_available_providers()]
    except Exception:
        return []


def _nested_set(data: dict[str, Any], parts: tuple[str, ...], value: Any) -> None:
    current = data
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _first_existing_path(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(value)
        if path.exists():
            return path
    return None


def _safe_filename(name: str) -> str:
    stem = Path(name).stem.replace(" ", "_")
    suffix = Path(name).suffix.lower()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "", stem).strip("._-") or "upload"
    return f"{cleaned[:80]}{suffix}"


def _format_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {item.strip().lower().lstrip(".") for item in value.split(",") if item.strip()}
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower().lstrip(".") for item in value if str(item).strip()}
    return set()


def _model_ids_from_response(response: Any) -> list[str]:
    if isinstance(response, dict):
        data = response.get("data")
    else:
        data = getattr(response, "data", None)
    if not isinstance(data, list):
        return []
    ids: list[str] = []
    for item in data:
        model_id = item.get("id") if isinstance(item, dict) else getattr(item, "id", None)
        if model_id:
            ids.append(str(model_id))
    return sorted(set(ids))


def _validate_cleanup_root(path: Path) -> None:
    resolved = path.resolve()
    if resolved in {PROJECT_ROOT, PROJECT_ROOT.parent, resolved.anchor}:
        raise ValueError(f"Refusing to clear unsafe jobs directory: {resolved}")
    protected_children = (
        PROJECT_ROOT / "apps",
        PROJECT_ROOT / "eistara",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "tests",
        PROJECT_ROOT / "docs",
        PROJECT_ROOT / ".venv",
        PROJECT_ROOT / "_model_cache",
        PROJECT_ROOT / "models",
    )
    for protected in protected_children:
        protected_resolved = protected.resolve()
        if resolved == protected_resolved or resolved.is_relative_to(protected_resolved):
            raise ValueError(f"Refusing to clear protected project directory: {resolved}")


def _all_jobs_reached(jobs, target_stage: StageName) -> bool:
    target_index = STAGE_ORDER.index(target_stage)
    for job in jobs:
        completed = set(job.state.completed_stages)
        reached = any(STAGE_ORDER.index(stage) >= target_index for stage in completed)
        if not reached:
            return False
    return True


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        result = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=False)
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
