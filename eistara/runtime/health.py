from __future__ import annotations

import shutil
import socket
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    name: str
    ok: bool
    detail: str
    kind: str = "generic"
    required: bool = True


@dataclass(frozen=True, slots=True)
class RuntimeHealthReport:
    checks: list[DependencyCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks if check.required)

    def by_kind(self, kind: str) -> list[DependencyCheck]:
        return [check for check in self.checks if check.kind == kind]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "detail": check.detail,
                    "kind": check.kind,
                    "required": check.required,
                }
                for check in self.checks
            ],
        }


ToolProbe = Callable[[str], bool]
UrlProbe = Callable[[str, float], bool]
LlmChatProbe = Callable[[str, str | None, str | None, bool, float, str | None, bool], tuple[bool, str]]


def default_tool_probe(tool: str) -> bool:
    return shutil.which(tool) is not None


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


def llm_chat_url(base_url: str) -> str:
    probe = base_url.strip().rstrip("/")
    if not probe:
        return ""
    if not probe.endswith("/chat/completions"):
        probe += "/chat/completions"
    return probe


def service_root_url(api_url: str) -> str:
    parsed = urlsplit(api_url if "://" in api_url else f"http://{api_url}")
    if not parsed.netloc:
        return api_url
    return f"{parsed.scheme}://{parsed.netloc}"


def default_llm_chat_probe(
    base_url: str,
    api_key: str | None,
    model: str | None,
    llm_support_json: bool,
    timeout: float = 10.0,
    proxy_url: str | None = None,
    trust_env_proxy: bool = True,
) -> tuple[bool, str]:
    key = str(api_key or "").strip()
    model_name = str(model or "").strip()
    if not key:
        return False, "missing API key"
    if not model_name:
        return False, "missing model"
    try:
        import requests

        from eistara.adapters.llm.openai_compatible import normalize_openai_base_url
    except Exception as exc:
        return False, f"probe unavailable: {exc}"

    url = llm_chat_url(normalize_openai_base_url(base_url))
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say ok."}],
    }
    if llm_support_json:
        payload["response_format"] = {"type": "json_object"}
        payload["messages"][0]["content"] = "Return JSON only with a boolean field named ok set to true."
    try:
        proxy = str(proxy_url or "").strip()
        proxies = {"http": proxy, "https": proxy} if proxy else None
        session = requests.Session()
        session.trust_env = bool(trust_env_proxy)
        response = session.post(
            url,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "curl/8.19.0",
            },
            json=payload,
            timeout=timeout,
            proxies=proxies,
        )
    except Exception as exc:
        return False, f"{url}: connection failure: {exc}"
    if response.status_code >= 400:
        preview = (response.text or "")[:160].replace("\n", " ")
        return False, f"{url}: HTTP {response.status_code}: {preview}"
    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as exc:
        return False, f"{url}: invalid chat response: {exc}"
    if not str(content or "").strip():
        return False, f"{url}: empty chat response"
    return True, f"{url}: chat ok"


class RuntimeHealthService:
    def __init__(
        self,
        tool_probe: ToolProbe = default_tool_probe,
        url_probe: UrlProbe = default_url_probe,
        llm_chat_probe: LlmChatProbe = default_llm_chat_probe,
        timeout_sec: float = 2.0,
        llm_chat_timeout_sec: float = 30.0,
    ):
        self.tool_probe = tool_probe
        self.url_probe = url_probe
        self.llm_chat_probe = llm_chat_probe
        self.timeout_sec = timeout_sec
        self.llm_chat_timeout_sec = llm_chat_timeout_sec

    def check(
        self,
        *,
        llm_base_url: str | None = None,
        llm_api_key: str | None = None,
        llm_model: str | None = None,
        llm_support_json: bool = False,
        llm_proxy_url: str | None = None,
        llm_trust_env_proxy: bool = True,
        tts_api_url: str | None = None,
        tts_label: str = "TTS",
        tools: tuple[str, ...] = ("ffmpeg", "ffprobe"),
    ) -> RuntimeHealthReport:
        checks: list[DependencyCheck] = []
        for tool in tools:
            ok = self.tool_probe(tool)
            checks.append(
                DependencyCheck(
                    name=tool,
                    ok=ok,
                    detail="found" if ok else "not in PATH",
                    kind="tool",
                    required=True,
                )
            )

        if llm_base_url:
            url = llm_models_url(llm_base_url)
            ok = self.url_probe(url, self.timeout_sec)
            checks.append(DependencyCheck("LLM gateway", ok, url, kind="llm", required=False))
            has_key = bool(str(llm_api_key or "").strip())
            checks.append(
                DependencyCheck(
                    "LLM API key",
                    has_key,
                    "configured" if has_key else "missing",
                    kind="llm",
                    required=False,
                )
            )
            if has_key and str(llm_model or "").strip():
                chat_ok, chat_detail = self.llm_chat_probe(
                    llm_base_url,
                    llm_api_key,
                    llm_model,
                    llm_support_json,
                    self.llm_chat_timeout_sec,
                    llm_proxy_url,
                    llm_trust_env_proxy,
                )
                checks.append(DependencyCheck("LLM chat", chat_ok, chat_detail, kind="llm", required=False))

        if tts_api_url:
            url = service_root_url(tts_api_url)
            ok = self.url_probe(url, self.timeout_sec)
            checks.append(DependencyCheck(tts_label, ok, url, kind="tts", required=False))

        return RuntimeHealthReport(checks)
