from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eistara.core.jobs.models import StageName, utc_now_iso


@dataclass(slots=True)
class StageRecord:
    name: StageName
    status: str = "pending"
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    duration_sec: float | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    report: str | None = None
    log: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_sec": self.duration_sec,
            "outputs": self.outputs,
            "report": self.report,
            "log": self.log,
            "error": self.error,
        }


@dataclass(slots=True)
class Manifest:
    task_id: str
    workdir: str
    stage_order: list[StageName]
    schema_version: int = 1
    app: str = "Eistara"
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    input: dict[str, Any] = field(default_factory=dict)
    stages: dict[StageName, StageRecord] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    speakers: list[dict[str, Any]] = field(
        default_factory=lambda: [{"id": "SPEAKER_00", "label": "Default speaker", "role": "default"}]
    )
    warnings: list[str] = field(default_factory=list)
    caption_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "app": self.app,
            "task_id": self.task_id,
            "workdir": self.workdir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "input": self.input,
            "stage_order": [stage.value for stage in self.stage_order],
            "stages": {stage.value: record.to_dict() for stage, record in self.stages.items()},
            "outputs": self.outputs,
            "speakers": self.speakers,
            "warnings": self.warnings,
            "caption_source": self.caption_source,
        }
