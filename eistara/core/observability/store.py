from __future__ import annotations

import json
from pathlib import Path

from .events import JobEvent


EVENTS_FILE = "events.jsonl"
ARCHIVE_WORK_DIR = "work"


class JsonlEventStore:
    def __init__(self, jobs_dir: str | Path):
        self.jobs_dir = Path(jobs_dir).expanduser().resolve()

    def append(self, event: JobEvent) -> Path:
        path = self.job_events_path(event.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return path

    def read_job(self, job_id: str) -> list[JobEvent]:
        return self._read_path(self.job_events_path(job_id))

    def read_all(self) -> list[JobEvent]:
        events: list[JobEvent] = []
        if not self.jobs_dir.exists():
            return events
        pattern = f"*/{ARCHIVE_WORK_DIR}/{EVENTS_FILE}" if self.jobs_dir.name == "history" else f"*/{EVENTS_FILE}"
        for path in sorted(self.jobs_dir.glob(pattern)):
            events.extend(self._read_path(path))
        return sorted(events, key=lambda event: event.created_at)

    def job_events_path(self, job_id: str) -> Path:
        if self.jobs_dir.name == "history":
            return self.jobs_dir / job_id / ARCHIVE_WORK_DIR / EVENTS_FILE
        return self.jobs_dir / job_id / EVENTS_FILE

    def _read_path(self, path: Path) -> list[JobEvent]:
        if not path.exists():
            return []
        events: list[JobEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(JobEvent.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return events
