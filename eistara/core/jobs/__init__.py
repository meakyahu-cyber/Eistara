from .models import Job, JobState, JobStatus, StageName, STAGE_ORDER
from .store import JsonJobStore, history_dir_for_jobs

__all__ = ["Job", "JobFactory", "JobState", "JobStatus", "StageName", "STAGE_ORDER", "JsonJobStore", "history_dir_for_jobs", "read_tasks"]


def __getattr__(name: str):
    if name in {"JobFactory", "read_tasks"}:
        from .factory import JobFactory, read_tasks

        globals()["JobFactory"] = JobFactory
        globals()["read_tasks"] = read_tasks
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
