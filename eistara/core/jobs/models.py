from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eistara.compat.enum import StrEnum


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class StageName(StrEnum):
    DOWNLOAD = "download"
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"
    TTS_PREPARE = "tts_prepare"
    TTS = "tts"
    AUDIO_MIX = "audio_mix"
    QUALITY = "quality"
    COMPOSE = "compose"


STAGE_ORDER: tuple[StageName, ...] = (
    StageName.DOWNLOAD,
    StageName.TRANSCRIBE,
    StageName.TRANSLATE,
    StageName.TTS_PREPARE,
    StageName.TTS,
    StageName.AUDIO_MIX,
    StageName.COMPOSE,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class JobState:
    job_id: str
    status: JobStatus = JobStatus.PENDING
    current_stage: StageName | None = None
    completed_stages: list[StageName] = field(default_factory=list)
    failed_stage: StageName | None = None
    attempts: dict[StageName, int] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    error: str | None = None
    stage_started_at: str | None = None
    auto_requeue_count: int = 0
    stage_pid: int | None = None
    stage_run_token: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any], job_id: str) -> "JobState":
        def stage(value: Any) -> StageName | None:
            if not value:
                return None
            return StageName(str(value))

        def optional_int(value: Any) -> int | None:
            if value in (None, ""):
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        def non_negative_int(value: Any) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        return cls(
            job_id=str(data.get("job_id") or job_id),
            status=JobStatus(str(data.get("status") or JobStatus.PENDING)),
            current_stage=stage(data.get("current_stage")),
            completed_stages=[StageName(str(item)) for item in data.get("completed_stages") or []],
            failed_stage=stage(data.get("failed_stage")),
            attempts={StageName(str(key)): int(value) for key, value in (data.get("attempts") or {}).items()},
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            error=data.get("error"),
            stage_started_at=data.get("stage_started_at"),
            auto_requeue_count=non_negative_int(data.get("auto_requeue_count")),
            stage_pid=optional_int(data.get("stage_pid")),
            stage_run_token=str(data.get("stage_run_token")) if data.get("stage_run_token") else None,
            artifacts=dict(data.get("artifacts") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "completed_stages": [stage.value for stage in self.completed_stages],
            "failed_stage": self.failed_stage.value if self.failed_stage else None,
            "attempts": {stage.value: value for stage, value in self.attempts.items()},
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "stage_started_at": self.stage_started_at,
            "auto_requeue_count": self.auto_requeue_count,
            "stage_pid": self.stage_pid,
            "stage_run_token": self.stage_run_token,
            "artifacts": self.artifacts,
        }


@dataclass(frozen=True, slots=True)
class Job:
    job_id: str
    job_dir: Path
    task: dict[str, Any]
    state: JobState
