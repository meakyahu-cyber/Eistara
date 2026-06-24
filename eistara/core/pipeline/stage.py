from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from eistara.core.jobs.models import StageName


@dataclass(frozen=True, slots=True)
class StageContext:
    job_id: str
    job_dir: Path
    task: dict[str, Any]
    stage: StageName
    attempt: int
    config: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResult:
    status: str = "done"
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False


class StageRunner(Protocol):
    stage: StageName

    def run(self, context: StageContext) -> StageResult:
        """Run one pipeline stage for one job."""
