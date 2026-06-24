from __future__ import annotations

from pathlib import Path

from eistara.config.models import AppConfig
from eistara.runtime import dependency_report as dep
from eistara.runtime.dependency_report import build_model_dependency_report
from eistara.runtime.health import RuntimeHealthService, llm_chat_url, llm_models_url, service_root_url


def test_runtime_health_url_helpers() -> None:
    assert llm_models_url("https://example.test/v1") == "https://example.test/v1/models"
    assert llm_models_url("https://example.test/v1/models") == "https://example.test/v1/models"
    assert llm_chat_url("https://example.test/v1") == "https://example.test/v1/chat/completions"
    assert llm_chat_url("https://example.test/v1/chat/completions") == "https://example.test/v1/chat/completions"
    assert service_root_url("http://127.0.0.1:8010/tts") == "http://127.0.0.1:8010"


def test_runtime_health_required_tools_control_ok() -> None:
    service = RuntimeHealthService(
        tool_probe=lambda tool: tool == "ffmpeg",
        url_probe=lambda url, timeout: True,
    )

    report = service.check(tools=("ffmpeg", "ffprobe"))

    assert report.ok is False
    assert [check.name for check in report.by_kind("tool")] == ["ffmpeg", "ffprobe"]


def test_runtime_health_optional_services_do_not_fail_report() -> None:
    service = RuntimeHealthService(
        tool_probe=lambda tool: True,
        url_probe=lambda url, timeout: False,
    )

    report = service.check(llm_base_url="https://llm.test/v1", tts_api_url="http://tts.test/tts")

    assert report.ok is True
    assert report.by_kind("llm")[0].ok is False
    assert report.by_kind("llm")[1].detail == "missing"
    assert report.by_kind("tts")[0].detail == "http://tts.test"


def test_runtime_health_reports_configured_llm_key() -> None:
    service = RuntimeHealthService(
        tool_probe=lambda tool: True,
        url_probe=lambda url, timeout: True,
    )

    report = service.check(llm_base_url="https://llm.test/v1", llm_api_key="secret")

    assert report.by_kind("llm")[1].ok is True
    assert report.by_kind("llm")[1].detail == "configured"


def test_runtime_health_checks_llm_chat_when_model_is_configured() -> None:
    calls: list[tuple[str, str | None, str | None, bool, float, str | None, bool]] = []

    def chat_probe(
        base_url: str,
        api_key: str | None,
        model: str | None,
        llm_support_json: bool,
        timeout: float,
        proxy_url: str | None,
        trust_env_proxy: bool,
    ):
        calls.append((base_url, api_key, model, llm_support_json, timeout, proxy_url, trust_env_proxy))
        return False, "chat failed"

    service = RuntimeHealthService(
        tool_probe=lambda tool: True,
        url_probe=lambda url, timeout: True,
        llm_chat_probe=chat_probe,
        timeout_sec=3.0,
    )

    report = service.check(
        llm_base_url="https://llm.test/v1",
        llm_api_key="secret",
        llm_model="model-a",
        llm_support_json=True,
        llm_proxy_url="http://proxy.test:7890",
        llm_trust_env_proxy=False,
    )

    llm_checks = report.by_kind("llm")
    assert [check.name for check in llm_checks] == ["LLM gateway", "LLM API key", "LLM chat"]
    assert llm_checks[2].ok is False
    assert llm_checks[2].detail == "chat failed"
    assert calls == [("https://llm.test/v1", "secret", "model-a", True, 30.0, "http://proxy.test:7890", False)]


def test_runtime_health_to_dict() -> None:
    service = RuntimeHealthService(tool_probe=lambda tool: True, url_probe=lambda url, timeout: True)

    data = service.check(llm_base_url="https://llm.test/v1").to_dict()

    assert data["ok"] is True
    assert any(check["name"] == "LLM gateway" for check in data["checks"])


def test_model_dependency_report_marks_network_and_local_cache(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "_model_cache"
    whisper_cache = cache_root / "models--Systran--faster-whisper-large-v3" / "snapshots" / "x"
    whisper_cache.mkdir(parents=True)
    (whisper_cache / "model.bin").write_bytes(b"model")
    torch_cache = tmp_path / "torch" / "hub" / "checkpoints"
    torch_cache.mkdir(parents=True)
    demucs = torch_cache / "955717e8-8726e21a.th"
    demucs.write_bytes(b"demucs")
    (torch_cache / "wav2vec2_fairseq_base_ls960_asr_ls960.pth").write_bytes(b"align")
    monkeypatch.setattr(dep, "_torch_checkpoint_dir", lambda: torch_cache)
    monkeypatch.setattr(
        dep,
        "_package_version",
        lambda name: {
            "yt-dlp": "2026.6.9",
            "openai": "2.43.0",
            "whisperx": "3.8.6",
            "faster-whisper": "1.2.1",
            "demucs": "4.1.0a3",
            "spacy": "3.8.14",
        }.get(name, ""),
    )
    config = AppConfig.from_dict(
        {
            "api": {"base_url": "https://llm.test/v1", "key": "secret", "model": "gpt-5.5"},
            "indextts": {"api_url": "http://127.0.0.1:8010/tts"},
            "model_dir": cache_root.as_posix(),
            "whisper": {"runtime": "local", "model": "large-v3", "language": "en"},
            "demucs": {"enabled": True, "provider": "demucs"},
        }
    )

    report = build_model_dependency_report(config, project_root=tmp_path).to_dict()
    by_dependency = {item["dependency"]: item for item in report["items"]}

    assert report["ok"] is True
    assert by_dependency["LLM gateway"]["mode"] == "network"
    assert by_dependency["LLM gateway"]["ok"] is True
    assert by_dependency["WhisperX local ASR (large-v3)"]["status"] == "cached"
    assert by_dependency["Demucs htdemucs"]["path"] == str(demucs)
    assert by_dependency["TTS provider (indextts)"]["mode"] == "local-service"


def test_model_dependency_report_uses_belle_model_for_chinese_asr(monkeypatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "_model_cache"
    belle_cache = cache_root / "Belle-whisper-large-v3-zh-punct-fasterwhisper"
    belle_cache.mkdir(parents=True)
    (belle_cache / "model.bin").write_bytes(b"model")
    monkeypatch.setattr(dep, "_torch_checkpoint_dir", lambda: tmp_path / "torch" / "hub" / "checkpoints")
    monkeypatch.setattr(
        dep,
        "_package_version",
        lambda name: {
            "yt-dlp": "2026.6.9",
            "openai": "2.43.0",
            "whisperx": "3.8.6",
            "faster-whisper": "1.2.1",
            "demucs": "4.1.0a3",
            "spacy": "3.8.14",
        }.get(name, ""),
    )
    config = AppConfig.from_dict(
        {
            "api": {"base_url": "https://llm.test/v1", "key": "secret", "model": "gpt-5.5"},
            "indextts": {"api_url": "http://127.0.0.1:8010/tts"},
            "model_dir": cache_root.as_posix(),
            "whisper": {"runtime": "local", "model": "large-v3", "language": "zh"},
            "demucs": False,
        }
    )

    report = build_model_dependency_report(config, project_root=tmp_path).to_dict()
    by_dependency = {item["dependency"]: item for item in report["items"]}

    assert by_dependency["WhisperX local ASR (Belle-whisper-large-v3-zh-punct-fasterwhisper)"]["status"] == "cached"
