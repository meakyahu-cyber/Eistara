from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eistara.compat.enum import StrEnum
from eistara.core.jobs.models import StageName, utc_now_iso


class JobEventType(StrEnum):
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    STAGE_RETRYING = "stage_retrying"
    STAGE_FAILED = "stage_failed"
    JOB_RECOVERED = "job_recovered"
    JOB_RETRY_REQUESTED = "job_retry_requested"
    JOB_RESET_REQUESTED = "job_reset_requested"


@dataclass(frozen=True, slots=True)
class JobEvent:
    job_id: str
    event_type: JobEventType
    stage: StageName | None = None
    status: str = ""
    attempt: int | None = None
    duration_sec: float | None = None
    error: str | None = None
    outputs: dict[str, Any] = field(default_factory=dict)
    message: str = ""
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "job_id": self.job_id,
            "event_type": self.event_type.value,
            "stage": self.stage.value if self.stage else None,
            "status": self.status,
            "attempt": self.attempt,
            "duration_sec": self.duration_sec,
            "error": self.error,
            "outputs": self.outputs,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobEvent":
        return cls(
            job_id=str(data.get("job_id") or ""),
            event_type=JobEventType(str(data.get("event_type"))),
            stage=StageName(str(data["stage"])) if data.get("stage") else None,
            status=str(data.get("status") or ""),
            attempt=int(data["attempt"]) if data.get("attempt") is not None else None,
            duration_sec=float(data["duration_sec"]) if data.get("duration_sec") is not None else None,
            error=data.get("error"),
            outputs=dict(data.get("outputs") or {}),
            message=str(data.get("message") or ""),
            created_at=str(data.get("created_at") or utc_now_iso()),
        )
