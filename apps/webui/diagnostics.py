from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from eistara.core.jobs import Job


SECRET_KEY_PARTS = ("key", "api_key", "token", "secret", "password", "cookie", "authorization")
MEDIA_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".m4a", ".aac"}
EXCLUDED_INVENTORY_DIRS = {"diagnostics", "__pycache__"}
LOG_TAIL_BYTES = 128 * 1024
EVENT_TAIL_BYTES = 96 * 1024
SUMMARY_TEXT_LIMIT = 700
EVENT_TEXT_LIMIT = 420


def build_diagnostic_summary(
    *,
    job: Job,
    outputs: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
    quality_report: dict[str, Any] | None,
    events: list[dict[str, Any]],
    config_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = job.state.to_dict()
    output_rows = [
        {
            "role": str(item.get("role") or ""),
            "kind": str(item.get("kind") or ""),
            "filename": str(item.get("filename") or ""),
            "exists": bool(item.get("exists")),
            "size": int(item.get("size") or 0),
        }
        for item in outputs
        if str(item.get("kind") or "") != "internal"
    ]
    internal_rows = [
        {
            "role": str(item.get("role") or ""),
            "filename": str(item.get("filename") or ""),
            "exists": bool(item.get("exists")),
            "size": int(item.get("size") or 0),
        }
        for item in outputs
        if str(item.get("kind") or "") == "internal"
    ]
    return {
        "job_id": job.job_id,
        "job_dir": str(job.job_dir),
        "source": str(job.task.get("source") or ""),
        "title": str(job.task.get("title") or job.task.get("name") or ""),
        "status": state.get("status"),
        "current_stage": state.get("current_stage"),
        "failed_stage": state.get("failed_stage"),
        "completed_stages": state.get("completed_stages") or [],
        "attempts": state.get("attempts") or {},
        "created_at": state.get("created_at"),
        "updated_at": state.get("updated_at"),
        "error": _short_text(state.get("error"), SUMMARY_TEXT_LIMIT),
        "outputs": output_rows,
        "internal_artifacts": internal_rows,
        "missing_artifacts": _missing_artifacts(state.get("artifacts") or {}),
        "manifest_status": _manifest_status(manifest),
        "quality": _quality_summary(quality_report),
        "recent_events": _recent_events(events),
        "config": config_summary or {},
    }


def render_diagnostic_text(summary: dict[str, Any]) -> str:
    lines = [
        "Eistara diagnostic summary",
        f"job: {summary.get('job_id')}",
        f"status: {summary.get('status')}",
        f"stage: {summary.get('current_stage') or summary.get('failed_stage') or '-'}",
        f"source: {summary.get('source') or '-'}",
        f"updated: {summary.get('updated_at') or '-'}",
    ]
    if summary.get("error"):
        lines.append(f"error: {summary['error']}")
    config = summary.get("config") if isinstance(summary.get("config"), dict) else {}
    api = config.get("api") if isinstance(config.get("api"), dict) else {}
    if api:
        lines.append(
            "api: "
            f"base_url={api.get('base_url') or '-'}; "
            f"model={api.get('model') or '-'}; "
            f"key={'configured' if api.get('has_key') else 'missing'}; "
            f"proxy={api.get('proxy_url') or '-'}"
        )
    separation = config.get("vocal_separation") if isinstance(config.get("vocal_separation"), dict) else {}
    if separation:
        lines.append(
            "vocal_separation: "
            f"enabled={separation.get('enabled')}; "
            f"provider={separation.get('provider') or '-'}; "
            f"model={separation.get('audio_separator_model') or '-'}; "
            f"model_exists={separation.get('audio_separator_model_exists')}; "
            f"model_valid={separation.get('audio_separator_model_valid')}; "
            f"onnx_cuda={separation.get('onnx_cuda')}"
        )
    completed = summary.get("completed_stages") or []
    lines.append(f"completed_stages: {', '.join(completed) if completed else '-'}")
    attempts = summary.get("attempts") or {}
    lines.append(f"attempts: {json.dumps(attempts, ensure_ascii=False, sort_keys=True)}")

    outputs = summary.get("outputs") or []
    if outputs:
        lines.append("outputs:")
        for item in outputs:
            size = item.get("size") or 0
            lines.append(f"- {item.get('role')}: {item.get('filename')} ({item.get('kind')}, {size} bytes)")
    internal_artifacts = summary.get("internal_artifacts") or []
    if internal_artifacts:
        lines.append("internal_artifacts:")
        for item in internal_artifacts:
            size = item.get("size") or 0
            lines.append(f"- {item.get('role')}: {item.get('filename')} ({size} bytes)")

    missing = summary.get("missing_artifacts") or []
    if missing:
        lines.append("missing_artifacts:")
        for item in missing:
            lines.append(f"- {item.get('role')}: {item.get('path')}")

    manifest = summary.get("manifest_status") or {}
    if manifest:
        lines.append(f"manifest: {json.dumps(manifest, ensure_ascii=False, sort_keys=True)}")
    quality = summary.get("quality") or {}
    if quality:
        lines.append(f"quality: {json.dumps(quality, ensure_ascii=False, sort_keys=True)}")

    events = summary.get("recent_events") or []
    if events:
        lines.append("recent_events:")
        for item in events:
            lines.append(
                "- "
                f"{item.get('created_at')}: {item.get('event_type')} "
                f"stage={item.get('stage') or '-'} status={item.get('status') or '-'} "
                f"error={item.get('error') or '-'}"
            )
    return redact_text("\n".join(lines))


def build_diagnostic_package(
    *,
    job: Job,
    summary: dict[str, Any],
    config: dict[str, Any],
    config_source: Path | None = None,
    scheduler_log: Path | None = None,
) -> Path:
    destination = job.job_dir / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    package_path = destination / f"{_safe_name(job.job_id)}_diagnostic.zip"
    redacted_config = redact_mapping(config)
    inventory = file_inventory(job.job_dir)

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("diagnostic.txt", render_diagnostic_text(summary))
        archive.writestr("summary.json", json.dumps(redact_mapping(summary), ensure_ascii=False, indent=2, sort_keys=True))
        archive.writestr("file_inventory.json", json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True))
        archive.writestr("config.redacted.json", json.dumps(redacted_config, ensure_ascii=False, indent=2, sort_keys=True))
        if config_source:
            archive.writestr("config_source.txt", str(config_source))
        _write_small_json(archive, job.job_dir / "task.json", "task.json")
        _write_small_json(archive, job.job_dir / "state.json", "state.json")
        _write_small_json(archive, job.job_dir / "manifest.json", "manifest.json")
        _write_small_json(archive, job.job_dir / "output" / "quality_report.json", "quality_report.json")
        _write_tail(archive, job.job_dir / "events.jsonl", "events.tail.jsonl", EVENT_TAIL_BYTES)
        for log_path in sorted((job.job_dir / "logs").glob("*.log")):
            _write_tail(archive, log_path, f"logs/{log_path.stem}.tail.log", LOG_TAIL_BYTES)
        if scheduler_log and scheduler_log.exists():
            _write_tail(archive, scheduler_log, "scheduler.tail.log", LOG_TAIL_BYTES)
    return package_path


def redact_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_secret_key(key_text):
                redacted[key_text] = _secret_presence(item)
            else:
                redacted[key_text] = redact_mapping(item)
        return redacted
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def redact_text(text: str) -> str:
    redacted = re.sub(r"sk-[A-Za-z0-9_\-]{12,}", "sk-***REDACTED***", text)
    redacted = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{12,}", r"\1***REDACTED***", redacted)
    redacted = re.sub(r"(?i)(api[_-]?key|token|secret|password|authorization|cookie)(\s*[:=]\s*)([^\s,;]+)", r"\1\2***REDACTED***", redacted)
    return redacted


def tail_file(path: Path, max_bytes: int) -> str:
    try:
        with path.open("rb") as handle:
            if path.stat().st_size > max_bytes:
                handle.seek(-max_bytes, 2)
                data = handle.read()
                prefix = f"[tail truncated to {max_bytes} bytes]\n".encode("utf-8")
                data = prefix + data
            else:
                data = handle.read()
    except OSError:
        return ""
    return redact_text(data.decode("utf-8", errors="replace"))


def file_inventory(job_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not job_dir.exists():
        return rows
    for path in sorted(job_dir.rglob("*")):
        if any(part in EXCLUDED_INVENTORY_DIRS for part in path.relative_to(job_dir).parts):
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(job_dir).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        rows.append({"path": relative, "size": size, "kind": "media" if path.suffix.lower() in MEDIA_SUFFIXES else "file"})
    return rows


def _missing_artifacts(artifacts: dict[str, Any]) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for role, value in artifacts.items():
        if not isinstance(value, str) or not value:
            continue
        if not _looks_like_path(value):
            continue
        path = Path(value)
        if not path.exists():
            missing.append({"role": str(role), "path": value})
    return missing


def _manifest_status(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not manifest:
        return {}
    stages = manifest.get("stages") if isinstance(manifest.get("stages"), dict) else {}
    return {
        "caption_source": manifest.get("caption_source") or "",
        "stages": {
            str(name): {
                "status": record.get("status") if isinstance(record, dict) else "",
                "error": record.get("error") if isinstance(record, dict) else None,
            }
            for name, record in stages.items()
        },
    }


def _quality_summary(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {}
    keys = ("passed", "error_count", "warning_count", "issues")
    return {key: report.get(key) for key in keys if key in report}


def _recent_events(events: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for item in events[-limit:]:
        rows.append(
            {
                "created_at": item.get("created_at"),
                "event_type": item.get("event_type"),
                "stage": item.get("stage"),
                "status": item.get("status"),
                "attempt": item.get("attempt"),
                "error": _short_text(item.get("error"), EVENT_TEXT_LIMIT),
                "message": _short_text(item.get("message"), EVENT_TEXT_LIMIT),
            }
        )
    return rows


def _looks_like_path(value: str) -> bool:
    return "\\" in value or "/" in value or bool(Path(value).suffix)


def _short_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = redact_text(str(value)).replace("\r\n", "\n").replace("\r", "\n")
    text = _strip_embedded_log_dump(text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}... [truncated]"


def _strip_embedded_log_dump(text: str) -> str:
    markers = (
        "Output from ffmpeg/avlib:",
        "ffmpeg version ",
        "configuration: --",
        "Traceback (most recent call last):",
        "\n[in#",
        "\nlibavutil ",
    )
    cut_at = min((index for marker in markers if (index := text.find(marker)) >= 0), default=-1)
    if cut_at < 0:
        return text
    kept = text[:cut_at].strip()
    suffix = "[details in diagnostic package]"
    return f"{kept}\n{suffix}" if kept else suffix


def _write_small_json(archive: zipfile.ZipFile, source: Path, arcname: str) -> None:
    if not source.exists():
        return
    try:
        data = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return
    archive.writestr(arcname, json.dumps(redact_mapping(data), ensure_ascii=False, indent=2, sort_keys=True))


def _write_tail(archive: zipfile.ZipFile, source: Path, arcname: str, max_bytes: int) -> None:
    if source.exists():
        archive.writestr(arcname, tail_file(source, max_bytes))


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SECRET_KEY_PARTS)


def _secret_presence(value: Any) -> str:
    if value in (None, "", [], {}):
        return "***MISSING***"
    return "***REDACTED_CONFIGURED***"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "job"
