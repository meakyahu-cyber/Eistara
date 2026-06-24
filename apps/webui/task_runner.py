from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TaskRunner:
    state: str = "idle"
    current_step: int = -1
    total_steps: int = 0
    current_label: str = ""
    error_msg: str = ""

    _pause_event: threading.Event = field(default_factory=threading.Event)
    _stop_event: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None
    _steps: list[tuple[str, Callable[[], object]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._pause_event.set()

    @staticmethod
    def get(session_state, key: str) -> "TaskRunner":
        if key not in session_state:
            session_state[key] = TaskRunner()
        return session_state[key]

    def start(self, steps: list[tuple[str, Callable[[], object]]]) -> None:
        if self.state in {"running", "paused"}:
            return
        self._steps = steps
        self.total_steps = len(steps)
        self.current_step = -1
        self.current_label = ""
        self.error_msg = ""
        self.state = "running"
        self._pause_event.set()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def pause(self) -> None:
        if self.state == "running":
            self._pause_event.clear()
            self.state = "paused"

    def resume(self) -> None:
        if self.state == "paused":
            self._pause_event.set()
            self.state = "running"

    def stop(self) -> None:
        if self.state in {"running", "paused"}:
            self._stop_event.set()
            self._pause_event.set()
            self.state = "stopped"

    def reset(self) -> None:
        if self.state not in {"running", "paused"}:
            self.state = "idle"
            self.current_step = -1
            self.total_steps = 0
            self.current_label = ""
            self.error_msg = ""
            self._steps = []

    @property
    def is_active(self) -> bool:
        return self.state in {"running", "paused"}

    @property
    def is_done(self) -> bool:
        return self.state in {"completed", "stopped", "error"}

    @property
    def progress(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return min((self.current_step + 1) / self.total_steps, 1.0)

    def _run(self) -> None:
        try:
            for index, (label, func) in enumerate(self._steps):
                if self._stop_event.is_set():
                    self.state = "stopped"
                    return
                self._pause_event.wait()
                if self._stop_event.is_set():
                    self.state = "stopped"
                    return
                self.current_step = index
                self.current_label = label
                func()
            self.state = "completed"
        except Exception as exc:  # pragma: no cover - surfaced through Streamlit state.
            self.error_msg = str(exc)
            self.state = "error"
