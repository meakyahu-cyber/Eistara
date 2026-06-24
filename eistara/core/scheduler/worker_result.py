from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eistara.core.jobs import StageName
from eistara.core.pipeline import StageResult


@dataclass(frozen=True, slots=True)
class StageWorkerResult:
    job_id: str
    stage: StageName
    status: str = "done"
    outputs: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped: bool = False
    error: str | None = None
    traceback: str | None = None

    @classmethod
    def from_stage_result(cls, job_id: str, stage: StageName, result: StageResult) -> "StageWorkerResult":
        return cls(
            job_id=job_id,
            stage=stage,
            status=result.status,
            outputs=dict(result.outputs),
            warnings=list(result.warnings),
            skipped=result.skipped,
        )

    @classmethod
    def from_exception(cls, job_id: str, stage: StageName, error: str, traceback_text: str | None = None) -> "StageWorkerResult":
        return cls(
            job_id=job_id,
            stage=stage,
            status="exception",
            warnings=[error],
            error=error,
            traceback=traceback_text,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StageWorkerResult":
        return cls(
            job_id=str(data.get("job_id") or ""),
            stage=StageName(str(data.get("stage"))),
            status=str(data.get("status") or "done"),
            outputs=dict(data.get("outputs") or {}),
            warnings=[str(item) for item in data.get("warnings") or []],
            skipped=bool(data.get("skipped")),
            error=str(data.get("error")) if data.get("error") else None,
            traceback=str(data.get("traceback")) if data.get("traceback") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "stage": self.stage.value,
            "status": self.status,
            "outputs": self.outputs,
            "warnings": self.warnings,
            "skipped": self.skipped,
            "error": self.error,
            "traceback": self.traceback,
        }

    def to_stage_result(self) -> StageResult:
        return StageResult(
            status=self.status,
            outputs=dict(self.outputs),
            warnings=list(self.warnings),
            skipped=self.skipped,
        )


def write_stage_worker_result(path: str | Path, result: StageWorkerResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_stage_worker_result(path: str | Path) -> StageWorkerResult:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Worker result is not an object: {path}")
    return StageWorkerResult.from_dict(data)
