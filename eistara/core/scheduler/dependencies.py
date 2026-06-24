from __future__ import annotations

import time
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from eistara.config import AppConfig, ConfigLoader
from eistara.core.jobs import Job, StageName


UrlProbe = Callable[[str, float], bool]
LlmChatProbe = Callable[[str, str | None, str | None, bool, float, str | None, bool], tuple[bool, str]]
JOB_CONFIG_FILE = "config.yaml"


def default_url_probe(url: str, timeout: float = 2.0) -> bool:
    parsed = urlsplit(url if "://" in url else f"http://{url}")
    host = parsed.hostname
    if not host:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def llm_models_url(base_url: str) -> str:
    probe = base_url.strip().rstrip("/")
    if not probe:
        return ""
    if not probe.endswith("/models"):
        probe += "/models"
    return probe


def service_root_url(api_url: str) -> str:
    parsed = urlsplit(api_url if "://" in api_url else f"http://{api_url}")
    if not parsed.netloc:
        return api_url
    return f"{parsed.scheme}://{parsed.netloc}"


@dataclass(frozen=True, slots=True)
class DependencyTarget:
    label: str
    url: str
    kind: str = "url"
    api_key: str | None = None
    model: str | None = None
    llm_support_json: bool = False
    proxy_url: str | None = None
    trust_env_proxy: bool = True


@dataclass(slots=True)
class SchedulerDependencyProbe:
    enabled: bool = True
    ttl_sec: int = 30
    config: AppConfig | None = None
    url_probe: UrlProbe = default_url_probe
    llm_chat_probe: LlmChatProbe | None = None
    timeout_sec: float = 2.0
    llm_chat_timeout_sec: float = 30.0
    _cache: dict[str, tuple[float, bool, str]] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: AppConfig,
        *,
        url_probe: UrlProbe = default_url_probe,
        llm_chat_probe: LlmChatProbe | None = None,
        timeout_sec: float = 2.0,
        llm_chat_timeout_sec: float = 30.0,
    ) -> "SchedulerDependencyProbe":
        return cls(
            enabled=config.batch.dependency_probe,
            ttl_sec=max(1, int(config.batch.dependency_probe_ttl_sec)),
            config=config,
            url_probe=url_probe,
            llm_chat_probe=llm_chat_probe,
            timeout_sec=timeout_sec,
            llm_chat_timeout_sec=llm_chat_timeout_sec,
        )

    def ready(self, job: Job, stage: StageName) -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        for target in self.targets_for(job, stage):
            ok, detail = self._cached_target_ok(target)
            if not ok:
                return False, f"{target.label} unreachable ({detail})"
        return True, ""

    def targets_for(self, job: Job, stage: StageName) -> tuple[DependencyTarget, ...]:
        config = self._job_config(job.job_dir)
        if config is None:
            return ()
        if stage == StageName.TRANSLATE:
            base_url = str(config.api.base_url or "").strip()
            if not base_url:
                return ()
            api_key = str(config.api.key or "").strip()
            model = str(config.api.model or "").strip()
            if self.llm_chat_probe and api_key and model:
                return (
                    DependencyTarget(
                        "LLM chat",
                        base_url,
                        kind="llm_chat",
                        api_key=api_key,
                        model=model,
                        llm_support_json=bool(config.api.llm_support_json),
                        proxy_url=str(config.api.proxy_url or "").strip() or None,
                        trust_env_proxy=bool(config.api.trust_env_proxy),
                    ),
                )
            return (DependencyTarget("LLM gateway", llm_models_url(base_url)),)

        if stage not in {StageName.TTS_PREPARE, StageName.TTS}:
            return ()

        method = str(config.tts_method or "").strip()
        provider_config = config.tts_backend_config(method)
        api_url = str(provider_config.get("api_url") or "").strip()
        if not api_url and method == "custom_tts":
            api_url = str(config.indextts.get("api_url") or "").strip()
        if not api_url:
            return ()
        return (DependencyTarget(f"TTS ({method})", service_root_url(api_url)),)

    def _cached_target_ok(self, target: DependencyTarget) -> tuple[bool, str]:
        now = time.time()
        cache_key = self._cache_key(target)
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0]) < self.ttl_sec:
            return cached[1], cached[2]
        ok, detail = self._probe_target(target)
        self._cache[cache_key] = (now, ok, detail)
        return ok, detail

    def _probe_target(self, target: DependencyTarget) -> tuple[bool, str]:
        if target.kind == "llm_chat" and self.llm_chat_probe is not None:
            ok, detail = self.llm_chat_probe(
                target.url,
                target.api_key,
                target.model,
                target.llm_support_json,
                self.llm_chat_timeout_sec,
                target.proxy_url,
                target.trust_env_proxy,
            )
            return ok, detail
        ok = self.url_probe(target.url, self.timeout_sec)
        return ok, target.url

    def _cache_key(self, target: DependencyTarget) -> str:
        return "|".join(
            [
                target.kind,
                target.url,
                str(target.model or ""),
                "json" if target.llm_support_json else "text",
                "env-proxy" if target.trust_env_proxy else "direct",
            ]
        )

    def _job_config(self, job_dir: Path) -> AppConfig | None:
        if self.config is not None:
            return self.config
        job_config = job_dir / JOB_CONFIG_FILE
        if job_config.exists():
            return ConfigLoader(job_config).load()
        return None
