from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from eistara.core.jobs import StageName
from eistara.core.pipeline import StageResult


@dataclass(frozen=True, slots=True)
class StageDiagnosticContext:
    job_id: str
    job_dir: Path
    task: dict[str, Any]
    artifacts: dict[str, Any]
    stage: StageName
    attempt: int
    duration_sec: float
    log_path: Path | None = None


class DiagnosticsHook(Protocol):
    def on_stage_finished(self, context: StageDiagnosticContext, result: StageResult) -> None:
        """Observe a completed stage result."""

    def on_stage_failed(self, context: StageDiagnosticContext, error: str, result: StageResult | None = None) -> None:
        """Observe a failed stage result or exception."""
