from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


LOCK_FILE = "scheduler.lock"


def is_pid_running(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def read_lock_pid(lock_path: Path) -> int | None:
    try:
        text = lock_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    match = re.search(r"^pid=(\d+)\s*$", text, flags=re.MULTILINE)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class SchedulerLock:
    def __init__(self, jobs_dir: str | os.PathLike[str], clear_lock: bool = False):
        self.path = Path(jobs_dir).expanduser().resolve() / LOCK_FILE
        self.clear_lock = clear_lock
        self.handle: int | None = None

    def remove_stale_if_needed(self) -> bool:
        if not self.path.exists():
            return False
        pid = read_lock_pid(self.path)
        if pid and is_pid_running(pid):
            return False
        try:
            self.path.unlink()
            return True
        except FileNotFoundError:
            return True
        except OSError:
            return False

    def __enter__(self) -> "SchedulerLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.clear_lock and self.path.exists():
            self.path.unlink()
        self.remove_stale_if_needed()
        try:
            self.handle = os.open(os.fspath(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise RuntimeError(f"Scheduler is already locked: {self.path}") from exc
        os.write(self.handle, f"pid={os.getpid()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.handle is not None:
            os.close(self.handle)
            self.handle = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
