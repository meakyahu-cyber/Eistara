from __future__ import annotations

import json
from pathlib import Path

import pytest

from apps.webui import backend as webui_backend
from apps.webui.backend import WebUiBackend, WebUiSettings
from apps.webui.main import STAGE_OPTIONS, _model_options
from eistara.config.loader import load_mapping
from eistara.core.jobs import STAGE_ORDER, JobStatus, JsonJobStore, history_dir_for_jobs
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.runtime import WEBUI_DEFAULT_PRESET


def write_job(jobs_dir: Path) -> Path:
    job_dir = jobs_dir / "job_0001_webui"
    job_dir.mkdir(parents=True)
    (job_dir / TASK_FILE).write_text(json.dumps({"id": job_dir.name, "source": "demo.mp4"}), encoding="utf-8")
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": [],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    return job_dir


def test_webui_backend_dashboard_counts_jobs(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    dashboard = backend.dashboard()

    assert dashboard["counts"] == {JobStatus.PENDING.value: 1}
    assert dashboard["jobs"][0]["job"] == "job_0001_webui"


def test_webui_backend_job_detail_reads_state_and_events(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    detail = backend.job_detail("job_0001_webui")

    assert detail["task"]["source"] == "demo.mp4"
    assert detail["state"]["status"] == "pending"
    assert detail["events"] == []


def test_webui_backend_marks_machine_json_outputs_internal(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    output_dir = job_dir / "output"
    internal_dir = output_dir / "internal"
    internal_dir.mkdir(parents=True)
    (output_dir / "src.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhello\n", encoding="utf-8")
    (internal_dir / "translations.json").write_text('{"translations":[]}', encoding="utf-8")
    (internal_dir / "subtitle_rows.json").write_text('{"rows":[]}', encoding="utf-8")
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["artifacts"] = {
        "source_srt": str(output_dir / "src.srt"),
        "translations_json": str(internal_dir / "translations.json"),
        "subtitle_rows_json": str(internal_dir / "subtitle_rows.json"),
    }
    (job_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    outputs = backend.job_outputs("job_0001_webui")

    kinds = {item["role"]: item["kind"] for item in outputs}
    assert kinds["source_subtitle"] == "subtitle"
    assert kinds["translations"] == "internal"
    assert kinds["subtitle_rows"] == "internal"


def test_webui_dashboard_exposes_short_redacted_error_summary(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["status"] = "failed"
    state["failed_stage"] = "translate"
    state["error"] = "api_key=sk-raw-secret-token\nTraceback (most recent call last):\nvery long internal stack"
    (job_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    row = backend.active_dashboard()["jobs"][0]

    assert row["error"]
    assert "sk-raw-secret-token" not in row["error_summary"]
    assert "Traceback" not in row["error_summary"]
    assert "api_key=***REDACTED***" in row["error_summary"]


def test_webui_backend_run_once_uses_production_preset(monkeypatch, tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    captured: dict[str, str] = {}

    class FakeScheduler:
        def run_once_with_lock(self, *, clear_lock: bool):
            captured["clear_lock"] = str(clear_lock)
            return True

    def fake_build_scheduler(jobs_dir, *, preset, config, max_stage_retries):
        captured["preset"] = preset
        return FakeScheduler()

    monkeypatch.setattr(webui_backend, "build_scheduler", fake_build_scheduler)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    result = backend.run_once()

    assert result == {"ran": True, "preset": "production"}
    assert captured["preset"] == "production"


def test_webui_backend_creates_jobs_from_sources(tmp_path: Path) -> None:
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs"))

    result = backend.create_jobs_from_sources(str(source))

    assert result["created"] == 1
    assert len(result["jobs"]) == 1
    assert result["jobs"][0].isdigit()
    assert len(result["jobs"][0]) == 10
    assert (tmp_path / "jobs" / result["jobs"][0] / TASK_FILE).exists()


def test_webui_single_job_disables_auto_archive(tmp_path: Path) -> None:
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs"))

    result = backend.create_single_job(str(source))
    job = JsonJobStore(tmp_path / "jobs").load(result["job"])

    assert job.task["archive_on_done"] is False


def test_webui_backend_archives_completed_single_job_manually(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["status"] = "done"
    state["completed_stages"] = [stage.value for stage in STAGE_ORDER]
    state["artifacts"] = {"source_video": str(job_dir / "output" / "Demo Video.mp4")}
    (job_dir / "output").mkdir()
    (job_dir / "output" / "Demo Video.mp4").write_bytes(b"video")
    (job_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    result = backend.archive_job("job_0001_webui")

    assert result["job"] == "Demo Video"
    assert not job_dir.exists()
    assert (history_dir_for_jobs(jobs_dir) / "Demo Video" / STATE_FILE).exists()


def test_webui_backend_deletes_only_requested_active_job(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    first = write_job(jobs_dir)
    second = jobs_dir / "job_0002_webui"
    second.mkdir(parents=True)
    (second / TASK_FILE).write_text(json.dumps({"id": second.name, "source": "second.mp4"}), encoding="utf-8")
    (second / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": second.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": [],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    result = backend.delete_active_job(first.name)

    assert result["deleted"] is True
    assert not first.exists()
    assert second.exists()


def test_webui_clear_jobs_refuses_project_source_directories() -> None:
    for name in ("apps", "eistara", "tests", "scripts", ".venv", "_model_cache", "models"):
        backend = WebUiBackend(WebUiSettings(webui_backend.PROJECT_ROOT / name))

        with pytest.raises(ValueError, match="protected project directory|unsafe jobs directory"):
            backend.clear_jobs()


def test_webui_latest_active_job_ignores_archived_history(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    archived_dir = write_job(jobs_dir)
    archived_state = json.loads((archived_dir / STATE_FILE).read_text(encoding="utf-8"))
    archived_state["status"] = "done"
    archived_state["completed_stages"] = [stage.value for stage in STAGE_ORDER]
    archived_state["artifacts"] = {"source_video": str(archived_dir / "output" / "Archived Demo.mp4")}
    (archived_dir / "output").mkdir()
    (archived_dir / "output" / "Archived Demo.mp4").write_bytes(b"video")
    (archived_dir / STATE_FILE).write_text(json.dumps(archived_state, ensure_ascii=False, indent=2), encoding="utf-8")
    JsonJobStore(jobs_dir).archive_done("job_0001_webui")
    active_dir = write_job(jobs_dir)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    assert backend.latest_active_job_id() == active_dir.name
    assert backend.latest_job_id(include_history=False) == active_dir.name
    assert backend.latest_job_id(include_history=True) in {active_dir.name, "Archived Demo"}


def test_webui_backend_splits_active_and_history_dashboards(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state = json.loads((job_dir / STATE_FILE).read_text(encoding="utf-8"))
    state["status"] = "done"
    state["completed_stages"] = [stage.value for stage in STAGE_ORDER]
    state["artifacts"] = {"source_video": str(job_dir / "output" / "History Demo.mp4")}
    (job_dir / "output").mkdir()
    (job_dir / "output" / "History Demo.mp4").write_bytes(b"video")
    (job_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    JsonJobStore(jobs_dir).archive_done("job_0001_webui")
    write_job(jobs_dir)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    active = backend.active_dashboard()
    history = backend.history_dashboard()
    combined = backend.dashboard()

    assert [row["location"] for row in active["jobs"]] == ["jobs"]
    assert [row["location"] for row in history["jobs"]] == ["history"]
    assert {row["location"] for row in combined["jobs"]} == {"jobs", "history"}


def test_webui_backend_health_includes_tts_section(tmp_path: Path) -> None:
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs"))

    health = backend.health()

    assert health["preset"] == WEBUI_DEFAULT_PRESET
    assert "tts" in health
    assert "method" in health["tts"]
    assert "checks" in health["tts"]
    assert "needs_recovery" in health["scheduler"]
    assert "model_dependencies" in health
    assert "items" in health["model_dependencies"]
    assert "youtube_cookies" in health


def test_webui_backend_health_does_not_probe_llm_by_default(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class FakeHealthReport:
        def to_dict(self):
            return {"ok": True, "checks": []}

    class FakeHealthService:
        def check(self, **kwargs):
            captured.update(kwargs)
            return FakeHealthReport()

    monkeypatch.setattr(webui_backend, "RuntimeHealthService", FakeHealthService)
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs"))

    backend.health()

    assert captured["llm_base_url"] is None
    assert captured["llm_api_key"] is None
    assert captured["llm_model"] is None


def test_webui_backend_health_is_cached_per_backend_instance(monkeypatch, tmp_path: Path) -> None:
    calls = 0

    class FakeHealthReport:
        def to_dict(self):
            return {"ok": True, "checks": []}

    class FakeHealthService:
        def check(self, **kwargs):
            nonlocal calls
            calls += 1
            return FakeHealthReport()

    monkeypatch.setattr(webui_backend, "RuntimeHealthService", FakeHealthService)
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs"))

    first = backend.health()
    second = backend.health()

    assert first is second
    assert calls == 1


def test_webui_backend_job_detail_can_skip_heavy_sections(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    detail = backend.job_detail("job_0001_webui", include_events=False, include_reports=False)

    assert detail["events"] == []
    assert detail["manifest"] is None
    assert detail["quality_report"] is None


def test_webui_settings_defaults_to_native_production(tmp_path: Path) -> None:
    settings = WebUiSettings(tmp_path / "jobs")

    assert settings.preset == "production"


def test_webui_stage_options_follow_v1_stage_order() -> None:
    assert STAGE_OPTIONS == [stage.value for stage in STAGE_ORDER]


def test_webui_backend_scheduler_safety_reports_running_jobs(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "running"
    state["current_stage"] = "translate"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    safety = backend.scheduler_safety()

    assert safety["running_count"] == 1
    assert safety["needs_recovery"] is True


def test_webui_backend_retries_failed_job(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["status"] = "failed"
    state["failed_stage"] = "download"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    result = backend.retry_failed("job_0001_webui")

    assert result["status"] == "pending"
    assert result["failed_stage"] == "download"


def test_webui_backend_resets_job_from_stage(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = write_job(jobs_dir)
    state_path = job_dir / STATE_FILE
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["completed_stages"] = ["download", "transcribe"]
    state["artifacts"] = {"source_video": "source.mp4", "subtitle_rows_json": "rows.json"}
    state_path.write_text(json.dumps(state), encoding="utf-8")
    backend = WebUiBackend(WebUiSettings(jobs_dir))

    result = backend.reset_from_stage("job_0001_webui", "transcribe")

    assert result["status"] == "pending"
    assert result["completed_stages"] == ["download"]


def test_webui_backend_configures_youtube_cookies(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs", config_path=config_path))

    result = backend.configure_youtube_cookies(browser="firefox", profile="default-release")
    data = load_mapping(config_path)

    assert result["updated"] is True
    assert data["youtube"]["cookies_from_browser"] == "firefox"
    assert data["source"]["cookies_browser_profile"] == "default-release"


def test_webui_model_ids_from_openai_models_response() -> None:
    response = {
        "object": "list",
        "data": [
            {"id": "claude-opus-4-8"},
            {"id": "gpt-4.1"},
            {"id": "claude-opus-4-8"},
            {"object": "model"},
        ],
    }

    assert webui_backend._model_ids_from_response(response) == ["claude-opus-4-8", "gpt-4.1"]


def test_webui_model_options_keep_current_model_first_without_duplicates() -> None:
    assert _model_options("claude-opus-4-8", ["gpt-4.1", "claude-opus-4-8"]) == [
        "claude-opus-4-8",
        "gpt-4.1",
    ]


def test_webui_default_config_updates_project_local_config(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "config.local.yaml"
    monkeypatch.setattr(webui_backend, "DEFAULT_LOCAL_CONFIG", config_path)
    backend = WebUiBackend(WebUiSettings(tmp_path / "work" / "jobs"))

    result = backend.update_config({"api.model": "claude-opus-4-8"})
    data = load_mapping(config_path)

    assert result["config_path"] == str(config_path)
    assert data["api"]["model"] == "claude-opus-4-8"
    assert not (tmp_path / "work" / "config.yaml").exists()


def test_webui_active_config_reports_write_path_and_api(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "api:\n"
        "  base_url: https://llm.test/v1\n"
        "  key: secret\n"
        "  model: model-a\n"
        "  llm_support_json: false\n"
        "  proxy_url: ''\n",
        encoding="utf-8",
    )
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs", config_path=config_path))

    active = backend.active_config()

    assert active["path"] == str(config_path.resolve())
    assert active["exists"] is True
    assert active["api"] == {
        "base_url": "https://llm.test/v1",
        "model": "model-a",
        "has_key": True,
        "llm_support_json": False,
        "proxy_url": "",
        "trust_env_proxy": False,
    }


def test_webui_reports_vocal_separation_model_status(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "demucs:\n"
        "  enabled: true\n"
        "  segment_minutes: 9\n",
        encoding="utf-8",
    )
    backend = WebUiBackend(WebUiSettings(tmp_path / "jobs", config_path=config_path))

    status = backend.vocal_separation_status()

    assert set(status) == {"enabled", "provider", "segment_minutes"}
    assert status["enabled"] is True
    assert status["provider"] == "demucs"
    assert status["segment_minutes"] == 9


def test_webui_diagnostic_summary_includes_vocal_separation(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    write_job(jobs_dir)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "demucs:\n"
        "  enabled: true\n"
        "  segment_minutes: 9\n",
        encoding="utf-8",
    )
    backend = WebUiBackend(WebUiSettings(jobs_dir, config_path=config_path))

    summary = backend.job_diagnostic_summary("job_0001_webui")
    text = backend.job_diagnostic_text("job_0001_webui")

    assert summary["config"]["vocal_separation"] == {
        "enabled": True,
        "provider": "demucs",
        "segment_minutes": 9,
    }
    assert "vocal_separation: enabled=True; provider=demucs; segment_minutes=9" in text


def test_webui_single_job_runs_until_requested_stage(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "demo.mp4"
    source.write_bytes(b"video")
    backend = WebUiBackend(WebUiSettings(tmp_path / "single_jobs"))
    created = backend.create_single_job(str(source), source_language="en", target_language="Simplified Chinese")

    class FakeScheduler:
        def __init__(self, jobs_dir: Path):
            self.job_store = webui_backend.JsonJobStore(jobs_dir)

        def recover_stalled_jobs(self):
            return 0

        def auto_requeue_failed_jobs(self):
            return 0

        def run_one_ready_stage(self):
            job = self.job_store.load(created["job"])
            completed = list(job.state.completed_stages)
            next_stage = STAGE_ORDER[len(completed)]
            state = job.state.to_dict()
            state["completed_stages"] = [*(stage.value for stage in completed), next_stage.value]
            state["status"] = "pending"
            (job.job_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            return True

    def fake_build_scheduler(jobs_dir, *, preset, config, max_stage_retries):
        assert preset == "production"
        return FakeScheduler(Path(jobs_dir))

    monkeypatch.setattr(webui_backend, "build_scheduler", fake_build_scheduler)

    result = backend.run_until_stage("translate")
    detail = backend.job_detail(created["job"])

    assert result["status"] == "reached"
    assert detail["state"]["completed_stages"] == ["download", "transcribe", "translate"]
    assert detail["state"]["status"] == "pending"
