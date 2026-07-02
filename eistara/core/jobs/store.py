from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .models import Job, JobState, JobStatus, StageName, STAGE_ORDER, utc_now_iso


TASK_FILE = "task.json"
STATE_FILE = "state.json"
WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}

ARTIFACT_KEYS_BY_STAGE: dict[StageName, set[str]] = {
    StageName.DOWNLOAD: {"source_video", "source_type"},
    StageName.TRANSCRIBE: {
        "language",
        "raw_audio",
        "high_quality_audio",
        "vocal_audio",
        "background_audio",
        "cleaned_chunks",
        "split_by_nlp",
        "segments",
        "subtitle_rows",
        "subtitle_rows_json",
    },
    StageName.TRANSLATE: {
        "translation",
        "subtitles",
        "audio_script",
        "publish_source_lines",
        "source_srt",
        "translated_srt",
        "audio_source_srt",
        "audio_translated_srt",
        "publish_translate_report",
        "terminology_json",
        "translations",
        "translation_count",
        "translations_json",
        "translation_items",
        "tts_segments",
        "tts_segments_json",
        "tts_segments_count",
    },
    StageName.TTS_PREPARE: {
        "tts_segments",
        "tts_segments_json",
        "tts_segments_count",
        "tts_tasks",
        "reference_audio_dir",
        "micro_tts_line_merge_report",
    },
    StageName.TTS: {"tts_outputs", "tts_count", "tts_durations", "tts_audio_quality_report"},
    StageName.AUDIO_MIX: {
        "audio_mix_plan",
        "dub_segments_json",
        "dub_audio",
        "dub_subtitles",
        "publish_retime_report",
        "clip_count",
        "audio_render_command",
        "audio_render_returncode",
    },
    StageName.QUALITY: {"quality_report", "passed", "error_count", "warning_count", "issues"},
    StageName.COMPOSE: {"compose_plan", "dub_video", "video_render_command", "video_render_returncode"},
}

INLINE_HEAVY_ARTIFACT_KEYS = {
    "segments",
    "subtitle_rows",
    "translations",
    "translation_items",
    "tts_segments",
    "tts_outputs",
    "tts_durations",
    "audio_render_command",
    "video_render_command",
}

MAX_INLINE_ARTIFACT_JSON_CHARS = 4000


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


def _state_to_persisted_dict(state: JobState) -> dict[str, Any]:
    data = state.to_dict()
    data["artifacts"] = _compact_state_artifacts(state.artifacts)
    return data


def _compact_state_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {}
    omitted: list[str] = []
    for key, value in artifacts.items():
        if key == "_omitted_inline_artifacts":
            continue
        if key in INLINE_HEAVY_ARTIFACT_KEYS:
            _add_artifact_count(compacted, key, value)
            omitted.append(key)
            continue
        if _is_small_state_artifact(value):
            compacted[key] = value
            continue
        _add_artifact_count(compacted, key, value)
        omitted.append(key)
    if omitted:
        compacted["_omitted_inline_artifacts"] = omitted
    return compacted


def _is_small_state_artifact(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return False
    return len(text) <= MAX_INLINE_ARTIFACT_JSON_CHARS


def _add_artifact_count(target: dict[str, Any], key: str, value: Any) -> None:
    count_key = f"{key}_count"
    if count_key in target:
        return
    if isinstance(value, (list, tuple, set, dict)):
        target[count_key] = len(value)


def history_dir_for_jobs(jobs_dir: str | os.PathLike[str]) -> Path:
    return Path(jobs_dir).expanduser().resolve().parent / "history"


def safe_job_name(text: str, fallback: str = "job") -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(text or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" ._")
    if not name:
        name = fallback
    if name.upper() in WINDOWS_RESERVED_NAMES:
        name = f"{name}_job"
    return name[:120].rstrip(" ._") or fallback


class JsonJobStore:
    """Filesystem-backed job store.

    This is intentionally small. The scheduler and WebUI should depend on this
    contract, so storage can later swap JSON for SQLite without rewriting business
    logic.
    """

    def __init__(self, jobs_dir: str | os.PathLike[str]):
        self.jobs_dir = Path(jobs_dir).expanduser().resolve()

    def discover(self) -> list[Job]:
        if not self.jobs_dir.exists():
            return []
        jobs: list[Job] = []
        for job_dir in sorted(path for path in self.jobs_dir.iterdir() if (path / TASK_FILE).exists()):
            jobs.append(self.load(job_dir.name))
        return jobs

    def load(self, job_id: str) -> Job:
        job_dir = self.jobs_dir / job_id
        return self.load_from_dir(job_dir, job_id=job_id)

    def load_from_dir(self, job_dir: str | os.PathLike[str], *, job_id: str | None = None) -> Job:
        job_dir = Path(job_dir).expanduser().resolve()
        job_id = job_id or job_dir.name
        task = _read_json(job_dir / TASK_FILE, {})
        state_data = _read_json(job_dir / STATE_FILE, None)
        if not isinstance(state_data, dict):
            state = JobState(job_id=job_id)
            _write_json_atomic(job_dir / STATE_FILE, state.to_dict())
        else:
            state = JobState.from_dict(state_data, job_id=job_id)
        return Job(job_id=job_id, job_dir=job_dir, task=task, state=state)

    def save_state(self, job_id: str, state: JobState) -> None:
        state.updated_at = utc_now_iso()
        _write_json_atomic(self.jobs_dir / job_id / STATE_FILE, _state_to_persisted_dict(state))

    def next_stage(self, state: JobState) -> StageName | None:
        completed = set(state.completed_stages)
        for stage in STAGE_ORDER:
            if stage not in completed:
                return stage
        return None

    def mark_running(
        self,
        job_id: str,
        stage: StageName,
        *,
        stage_pid: int | None = None,
        stage_run_token: str | None = None,
    ) -> JobState:
        job = self.load(job_id)
        state = job.state
        state.status = JobStatus.RUNNING
        state.current_stage = stage
        state.failed_stage = None
        state.error = None
        state.stage_started_at = utc_now_iso()
        state.stage_pid = stage_pid
        state.stage_run_token = stage_run_token
        state.attempts[stage] = int(state.attempts.get(stage, 0)) + 1
        self.save_state(job_id, state)
        return state

    def mark_done(self, job_id: str, stage: StageName, artifacts: dict[str, Any] | None = None) -> JobState:
        job = self.load(job_id)
        state = job.state
        if stage not in state.completed_stages:
            state.completed_stages.append(stage)
        if artifacts:
            state.artifacts.update(artifacts)
        state.current_stage = None
        state.stage_started_at = None
        state.stage_pid = None
        state.stage_run_token = None
        state.error = None
        state.status = JobStatus.DONE if self.next_stage(state) is None else JobStatus.PENDING
        self.save_state(job_id, state)
        return state

    def archive_done(self, job_id: str, history_dir: str | os.PathLike[str] | None = None) -> Path | None:
        job = self.load(job_id)
        if job.state.status != JobStatus.DONE:
            return None
        source = job.job_dir
        if not source.exists():
            return None
        history_root = Path(history_dir).expanduser().resolve() if history_dir is not None else history_dir_for_jobs(self.jobs_dir)
        history_root.mkdir(parents=True, exist_ok=True)
        archive_name = _archive_name_for_job(job)
        target = _unique_archive_path(history_root / archive_name)
        old_root = source.resolve()
        shutil.move(os.fspath(source), os.fspath(target))
        self._rewrite_job_files_after_rename(
            target.resolve(),
            old_root,
            target.resolve(),
            old_job_id=job_id,
            new_job_id=target.name,
        )
        _remove_redundant_archive_json(target)
        return target

    def rename(self, job_id: str, desired_name: str) -> Job:
        source = self.jobs_dir / job_id
        if not source.exists():
            return self.load(job_id)
        new_id = safe_job_name(desired_name, fallback=job_id)
        target = _unique_sibling_path(self.jobs_dir / new_id, current=source)
        if target == source:
            return self.load(job_id)

        old_root = source.resolve()
        shutil.move(os.fspath(source), os.fspath(target))
        new_root = target.resolve()
        actual_id = target.name
        self._rewrite_job_files_after_rename(target, old_root, new_root, old_job_id=job_id, new_job_id=actual_id)
        return self.load(actual_id)

    def _rewrite_job_files_after_rename(
        self,
        job_dir: Path,
        old_root: Path,
        new_root: Path,
        *,
        old_job_id: str,
        new_job_id: str,
    ) -> None:
        task_path = job_dir / TASK_FILE
        task = _read_json(task_path, {})
        if isinstance(task, dict):
            task = _rewrite_paths(task, old_root, new_root)
            task["id"] = new_job_id
            if not str(task.get("title") or "").strip():
                task["title"] = new_job_id
            _write_json_atomic(task_path, task)

        state_path = job_dir / STATE_FILE
        state = _read_json(state_path, {})
        if isinstance(state, dict):
            state = _rewrite_paths(state, old_root, new_root)
            state["job_id"] = new_job_id
            _write_json_atomic(state_path, state)

        manifest_path = job_dir / "manifest.json"
        manifest = _read_json(manifest_path, {})
        if isinstance(manifest, dict):
            manifest = _rewrite_paths(manifest, old_root, new_root)
            manifest["task_id"] = new_job_id
            manifest["workdir"] = str(new_root)
            _write_json_atomic(manifest_path, manifest)

        events_path = job_dir / "events.jsonl"
        if events_path.exists():
            rewritten: list[str] = []
            for line in events_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    rewritten.append(line)
                    continue
                if event.get("job_id") == old_job_id:
                    event["job_id"] = new_job_id
                event = _rewrite_paths(event, old_root, new_root)
                rewritten.append(json.dumps(event, ensure_ascii=False))
            events_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")

        _rewrite_json_files_after_rename(job_dir, old_root, new_root, old_job_id=old_job_id, new_job_id=new_job_id)

    def mark_failed(self, job_id: str, stage: StageName, error: str) -> JobState:
        job = self.load(job_id)
        state = job.state
        state.status = JobStatus.FAILED
        state.current_stage = None
        state.failed_stage = stage
        state.stage_started_at = None
        state.stage_pid = None
        state.stage_run_token = None
        state.error = error
        self.save_state(job_id, state)
        return state

    def retry_failed(self, job_id: str) -> JobState:
        job = self.load(job_id)
        state = job.state
        if state.status != JobStatus.FAILED:
            raise ValueError(f"Job is not failed: {job_id}")
        state.status = JobStatus.PENDING
        state.current_stage = None
        state.error = None
        state.stage_started_at = None
        state.stage_pid = None
        state.stage_run_token = None
        self.save_state(job_id, state)
        return state

    def reset_from_stage(self, job_id: str, stage: StageName | str) -> JobState:
        reset_stage = StageName(str(stage))
        job = self.load(job_id)
        state = job.state
        if state.status == JobStatus.RUNNING:
            raise ValueError(f"Cannot reset running job: {job_id}")

        reset_index = STAGE_ORDER.index(reset_stage)
        reset_stages = set(STAGE_ORDER[reset_index:])
        state.completed_stages = [item for item in state.completed_stages if item not in reset_stages]
        for item in reset_stages:
            state.attempts.pop(item, None)
        artifact_keys = set().union(*(ARTIFACT_KEYS_BY_STAGE[item] for item in reset_stages))
        for key in artifact_keys:
            state.artifacts.pop(key, None)
        state.status = JobStatus.PENDING
        state.current_stage = None
        state.failed_stage = None
        state.error = None
        state.stage_started_at = None
        state.stage_pid = None
        state.stage_run_token = None
        self.save_state(job_id, state)
        return state

    def recover_interrupted(self, jobs: Iterable[Job] | None = None) -> list[Job]:
        recovered: list[Job] = []
        for job in jobs if jobs is not None else self.discover():
            if job.state.status != JobStatus.RUNNING:
                continue
            state = job.state
            state.status = JobStatus.PENDING
            state.error = "Recovered from interrupted scheduler run"
            state.current_stage = None
            state.stage_started_at = None
            state.stage_pid = None
            state.stage_run_token = None
            self.save_state(job.job_id, state)
            recovered.append(self.load(job.job_id))
        return recovered


def _unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = utc_now_iso().replace(":", "").replace("-", "").replace("+", "_").replace("T", "_")
    candidate = path.with_name(f"{path.name}_{suffix}")
    index = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}_{suffix}_{index}")
        index += 1
    return candidate


def _archive_name_for_job(job: Job) -> str:
    source_video = job.state.artifacts.get("source_video")
    if source_video:
        name = Path(str(source_video)).stem
        if name:
            return safe_job_name(name, fallback=job.job_id)
    title = str(job.task.get("title") or "").strip()
    if title:
        return safe_job_name(title, fallback=job.job_id)
    return safe_job_name(job.job_id)


def _unique_sibling_path(path: Path, *, current: Path | None = None) -> Path:
    path = path.resolve()
    if current is not None and path == current.resolve():
        return path
    if not path.exists():
        return path
    index = 2
    while True:
        candidate = path.with_name(f"{path.name}_{index}")
        if current is not None and candidate.resolve() == current.resolve():
            return candidate
        if not candidate.exists():
            return candidate
        index += 1


def _rewrite_paths(value: Any, old_root: Path, new_root: Path) -> Any:
    old_text = os.fspath(old_root)
    new_text = os.fspath(new_root)
    old_posix = old_root.as_posix()
    new_posix = new_root.as_posix()
    if isinstance(value, str):
        return value.replace(old_text, new_text).replace(old_posix, new_posix)
    if isinstance(value, list):
        return [_rewrite_paths(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_paths(item, old_root, new_root) for key, item in value.items()}
    return value


def _rewrite_json_files_after_rename(
    job_dir: Path,
    old_root: Path,
    new_root: Path,
    *,
    old_job_id: str,
    new_job_id: str,
) -> None:
    for path in job_dir.rglob("*.json"):
        data = _read_json(path, None)
        if data is None:
            continue
        rewritten = _rewrite_paths(data, old_root, new_root)
        if isinstance(rewritten, dict):
            if rewritten.get("job_id") == old_job_id:
                rewritten["job_id"] = new_job_id
            if rewritten.get("task_id") == old_job_id:
                rewritten["task_id"] = new_job_id
        if rewritten != data:
            _write_json_atomic(path, rewritten)


def _remove_redundant_archive_json(job_dir: Path) -> None:
    patterns = (
        "logs/*.result.json",
        "output/audio/tmp/*.cache.json",
    )
    for pattern in patterns:
        for path in job_dir.glob(pattern):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
