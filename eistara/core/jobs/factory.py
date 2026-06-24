from __future__ import annotations

import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from eistara.core.jobs.models import JobState, utc_now_iso
from eistara.core.jobs.store import STATE_FILE, TASK_FILE, _unique_sibling_path, _write_json_atomic
from eistara.core.manifest import JsonManifestStore


JOB_CONFIG_FILE = "config.yaml"

SOURCE_KEYS = ("source", "url", "video", "video_file", "Video File")
TITLE_KEYS = ("title", "name", "Title", "Name")
RESOLUTION_KEYS = ("resolution", "ytb_resolution", "Resolution")
SOURCE_LANG_KEYS = ("source_language", "Source Language")
TARGET_LANG_KEYS = ("target_language", "Target Language")


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def resolve_source(source: str, input_path: Path | None = None, project_root: Path | None = None) -> str:
    if is_url(source):
        return source
    path = Path(source).expanduser()
    if path.is_absolute():
        return os.fspath(path)

    root = project_root or Path.cwd()
    candidates: list[Path] = []
    if input_path:
        candidates.append((input_path.parent / path).resolve())
    candidates.append((root / "batch" / "input" / path).resolve())
    candidates.append((root / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return os.fspath(candidate)
    return os.fspath(candidates[0] if candidates else path.resolve())


def normalize_task(raw: dict[str, Any], index: int, input_path: Path | None = None, project_root: Path | None = None) -> dict[str, Any]:
    source = _first_value(raw, SOURCE_KEYS)
    if source is None:
        raise ValueError(f"Task {index + 1} is missing a source/url/video field")
    source = resolve_source(str(source).lstrip("\ufeff").strip(), input_path=input_path, project_root=project_root)
    source_type = str(raw.get("source_type") or ("url" if is_url(source) else "file")).strip().lower()
    if source_type == "file":
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Task {index + 1} source file not found: {source_path}")
        if source_path.stat().st_size <= 0:
            raise ValueError(f"Task {index + 1} source file is empty: {source_path}")

    task: dict[str, Any] = {
        "source": source,
        "source_type": source_type,
        "title": str(_first_value(raw, TITLE_KEYS) or "").strip(),
        "resolution": str(_first_value(raw, RESOLUTION_KEYS) or "").strip(),
    }
    source_language = _first_value(raw, SOURCE_LANG_KEYS)
    target_language = _first_value(raw, TARGET_LANG_KEYS)
    if source_language is not None:
        task["source_language"] = str(source_language).strip()
    if target_language is not None:
        task["target_language"] = str(target_language).strip()
    if "archive_on_done" in raw:
        task["archive_on_done"] = bool(raw.get("archive_on_done"))
    return task


def read_tasks(input_path: str | os.PathLike[str], project_root: Path | None = None) -> list[dict[str, Any]]:
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Task input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        raw_tasks = data.get("tasks", []) if isinstance(data, dict) else data
        if not isinstance(raw_tasks, list):
            raise ValueError("JSON task input must be a list or an object with a tasks list")
        rows = [dict(item) for item in raw_tasks]
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as file:
            rows = [dict(row) for row in csv.DictReader(file)]
    else:
        rows = []
        with path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                source = line.strip()
                if source and not source.startswith("#"):
                    rows.append({"source": source})
    return [normalize_task(task, index, input_path=path, project_root=project_root) for index, task in enumerate(rows)]


class JobFactory:
    def __init__(
        self,
        jobs_dir: str | os.PathLike[str],
        project_root: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
    ):
        self.jobs_dir = Path(jobs_dir).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve() if project_root else Path.cwd().resolve()
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self.manifest_store = JsonManifestStore()

    def create_from_file(self, input_path: str | os.PathLike[str]) -> list[Path]:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        tasks = read_tasks(input_path, project_root=self.project_root)
        return self.create_from_tasks(tasks)

    def create_from_tasks(self, tasks: list[dict[str, Any]]) -> list[Path]:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        existing = self._existing_sources()
        created: list[Path] = []

        for task in tasks:
            if task["source"] in existing:
                created.append(existing[task["source"]])
                continue

            job_id = self._job_id_for_task(task)
            job_dir = _unique_sibling_path(self.jobs_dir / job_id)
            job_id = job_dir.name
            job_dir.mkdir(parents=True, exist_ok=False)

            task = dict(task)
            task["id"] = job_id
            task["created_at"] = utc_now_iso()
            if self.config_path and self.config_path.exists():
                shutil.copy2(self.config_path, job_dir / JOB_CONFIG_FILE)
            _write_json_atomic(job_dir / TASK_FILE, task)
            _write_json_atomic(job_dir / STATE_FILE, JobState(job_id=job_id).to_dict())
            self.manifest_store.load_or_create(job_dir, task)
            (job_dir / "logs").mkdir(exist_ok=True)
            created.append(job_dir)
        return created

    def _existing_sources(self) -> dict[str, Path]:
        sources: dict[str, Path] = {}
        if not self.jobs_dir.exists():
            return sources
        for task_path in self.jobs_dir.glob(f"*/{TASK_FILE}"):
            try:
                task = json.loads(task_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            source = task.get("source")
            if source:
                sources[str(source)] = task_path.parent
        return sources

    def _job_id_for_task(self, task: dict[str, Any]) -> str:
        return datetime.now().strftime("%y%m%d%H%M")
