from __future__ import annotations

from pathlib import Path

from eistara.core.jobs import JsonJobStore

from .heartbeat import SchedulerHeartbeat
from .lock import LOCK_FILE, is_pid_running, read_lock_pid


def recover_orphaned_scheduler_state(jobs_dir: str | Path) -> int:
    root = Path(jobs_dir).expanduser().resolve()
    lock_path = root / LOCK_FILE
    lock_pid = read_lock_pid(lock_path)
    if lock_path.exists() and not is_pid_running(lock_pid):
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    heartbeat = SchedulerHeartbeat(root)
    hb = heartbeat.read()
    hb_pid = None
    if hb:
        try:
            hb_pid = int(hb.get("pid"))
        except (TypeError, ValueError):
            hb_pid = None
    if hb and not is_pid_running(hb_pid):
        heartbeat.clear()

    return len(JsonJobStore(root).recover_interrupted())
