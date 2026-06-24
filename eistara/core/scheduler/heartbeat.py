from __future__ import annotations

import json
import os
import time
from pathlib import Path

from eistara.core.jobs.store import _write_json_atomic
from eistara.core.jobs.models import utc_now_iso


HEARTBEAT_FILE = "scheduler.heartbeat.json"


class SchedulerHeartbeat:
    def __init__(self, jobs_dir: str | os.PathLike[str]):
        self.path = Path(jobs_dir).expanduser().resolve() / HEARTBEAT_FILE

    def write(self, active: list[str] | None = None) -> None:
        payload = {
            "pid": os.getpid(),
            "ts": time.time(),
            "updated_at": utc_now_iso(),
            "active": sorted(active or []),
        }
        _write_json_atomic(self.path, payload)

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    def read(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

    def age_sec(self) -> float | None:
        try:
            return time.time() - self.path.stat().st_mtime
        except OSError:
            return None
