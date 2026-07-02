from __future__ import annotations

import json
import zipfile
from pathlib import Path

from apps.webui.backend import WebUiBackend, WebUiSettings
from apps.webui.diagnostics import redact_mapping, redact_text
from eistara.core.jobs import JobStatus, StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.observability import JobEvent, JobEventType, JsonlEventStore


def _write_failed_job(jobs_dir: Path) -> Path:
    job_dir = jobs_dir / "job_0001_diag"
    output_dir = job_dir / "output"
    logs_dir = job_dir / "logs"
    output_dir.mkdir(parents=True)
    logs_dir.mkdir()
    (job_dir / TASK_FILE).write_text(json.dumps({"id": job_dir.name, "source": "https://youtu.be/demo"}), encoding="utf-8")
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": JobStatus.FAILED.value,
                "current_stage": None,
                "completed_stages": [StageName.DOWNLOAD.value, StageName.TRANSCRIBE.value],
                "failed_stage": StageName.TRANSLATE.value,
                "attempts": {StageName.TRANSLATE.value: 2},
                "error": "Gateway failed with api_key=sk-raw-secret-token",
                "artifacts": {
                    "source_video": str(output_dir / "source_video.mp4"),
                    "translated_srt": str(output_dir / "missing.srt"),
                },
            }
        ),
        encoding="utf-8",
    )
    (job_dir / "manifest.json").write_text(
        json.dumps({"caption_source": "youtube_subtitle", "stages": {"translate": {"status": "failed", "error": "bad"}}}),
        encoding="utf-8",
    )
    (output_dir / "quality_report.json").write_text(json.dumps({"passed": False, "error_count": 1}), encoding="utf-8")
    (output_dir / "source_video.mp4").write_bytes(b"video")
    (logs_dir / "translate.log").write_text("Authorization: Bearer sk-super-secret-token\nfailure\n", encoding="utf-8")
    (logs_dir / "translate.result.json").write_text(json.dumps({"not": "included"}), encoding="utf-8")
    JsonlEventStore(jobs_dir).append(
        JobEvent(
            job_dir.name,
            JobEventType.STAGE_FAILED,
            stage=StageName.TRANSLATE,
            status="failed",
            attempt=2,
            error="api_key: sk-event-secret-token",
        )
    )
    return job_dir


def test_webui_diagnostic_summary_has_failure_context_without_logs(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_failed_job(jobs_dir)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    summary = backend.job_diagnostic_summary("job_0001_diag")
    detail = backend.job_detail("job_0001_diag")

    assert summary["status"] == "failed"
    assert summary["failed_stage"] == "translate"
    assert summary["attempts"] == {"translate": 2}
    assert summary["missing_artifacts"] == [{"role": "translated_srt", "path": str(jobs_dir / "job_0001_diag" / "output" / "missing.srt")}]
    assert summary["manifest_status"]["stages"]["translate"]["status"] == "failed"
    assert "logs" not in detail


def test_webui_diagnostic_summary_separates_internal_artifacts(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = _write_failed_job(jobs_dir)
    internal_dir = job_dir / "output" / "internal"
    internal_dir.mkdir(parents=True)
    (internal_dir / "translations.json").write_text('{"translations":[]}', encoding="utf-8")
    (internal_dir / "tts_segments.json").write_text('{"segments":[]}', encoding="utf-8")
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["artifacts"]["translations_json"] = str(internal_dir / "translations.json")
    state["artifacts"]["tts_segments_json"] = str(internal_dir / "tts_segments.json")
    (job_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    summary = backend.job_diagnostic_summary("job_0001_diag")

    assert all(item["role"] != "translations" for item in summary["outputs"])
    assert {"role": "translations", "filename": "translations.json", "exists": True, "size": 19} in summary["internal_artifacts"]
    assert {"role": "tts_segments", "filename": "tts_segments.json", "exists": True, "size": 15} in summary["internal_artifacts"]


def test_webui_diagnostic_package_redacts_secrets_and_excludes_media(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    _write_failed_job(jobs_dir)
    config_path = tmp_path / "config.local.json"
    config_path.write_text(
        json.dumps(
            {
                "api": {"key": "sk-config-secret-token", "base_url": "https://example.test", "model": "gpt-5.5"},
                "youtube": {"cookies": "SID=secret-cookie"},
            }
        ),
        encoding="utf-8",
    )
    backend = WebUiBackend(WebUiSettings(jobs_dir, config_path=config_path))

    package = backend.build_job_diagnostic_package("job_0001_diag")

    with zipfile.ZipFile(package["path"]) as archive:
        names = set(archive.namelist())
        combined = "\n".join(archive.read(name).decode("utf-8", errors="replace") for name in names)

    assert "diagnostic.txt" in names
    assert "summary.json" in names
    assert "config.redacted.json" in names
    assert "task.json" in names
    assert "state.json" in names
    assert "manifest.json" in names
    assert "quality_report.json" in names
    assert "events.tail.jsonl" in names
    assert "logs/translate.tail.log" in names
    assert "output/source_video.mp4" not in names
    assert "logs/translate.result.json" not in names
    assert "sk-config-secret-token" not in combined
    assert "sk-super-secret-token" not in combined
    assert "sk-event-secret-token" not in combined
    assert "secret-cookie" not in combined
    assert "***REDACTED" in combined


def test_webui_diagnostic_redaction_helpers() -> None:
    redacted = redact_mapping({"api": {"key": "sk-1234567890abcdef"}, "nested": {"token": "abc"}, "value": "Bearer sk-abcdefghijklmnop"})

    assert redacted["api"]["key"] == "***REDACTED_CONFIGURED***"
    assert redacted["nested"]["token"] == "***REDACTED_CONFIGURED***"
    assert "sk-abcdefghijklmnop" not in redacted["value"]
    assert "sk-raw-secret-token" not in redact_text("api_key=sk-raw-secret-token")
