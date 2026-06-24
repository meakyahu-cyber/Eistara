from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Any, Protocol
from urllib.parse import urljoin, urlsplit, urlunsplit


class LlmAdapterError(RuntimeError):
    """Base error for LLM adapter failures."""


class LlmServiceError(LlmAdapterError):
    """Infrastructure failure, such as timeout or 5xx."""


class LlmRequestError(LlmAdapterError):
    """Invalid request, auth, model, or quota style failure."""


class HttpResponse(Protocol):
    status_code: int
    content: bytes
    text: str
    headers: Any

    def raise_for_status(self) -> None:
        """Raise for non-success HTTP status."""

    def json(self) -> Any:
        """Return decoded JSON."""


class HttpTransport(Protocol):
    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> HttpResponse:
        """HTTP GET."""

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> HttpResponse:
        """HTTP POST."""


class RequestsHttpTransport:
    def __init__(self, *, trust_env: bool = True, proxy_url: str = "") -> None:
        import requests

        self._session = requests.Session()
        self._session.trust_env = trust_env
        proxy = str(proxy_url or "").strip()
        if proxy:
            self._session.proxies.update({"http": proxy, "https": proxy})

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> HttpResponse:
        return self._session.get(url, headers=headers, timeout=timeout)

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> HttpResponse:
        return self._session.post(url, headers=headers, json=json, timeout=timeout)


_CACHE_LOCK = Lock()
_CACHE_MISS = object()


@dataclass(frozen=True, slots=True)
class OpenAICompatibleSettings:
    base_url: str
    model: str
    api_key: str = ""
    timeout_sec: float = 300.0
    temperature: float | None = None
    response_format_json: bool = True
    user_agent: str = "curl/8.19.0"
    trust_env_proxy: bool = True
    proxy_url: str = ""
    extra_headers: dict[str, str] = field(default_factory=dict)
    max_retries: int = 6
    retry_base_delay_sec: float = 4.0
    retry_max_delay_sec: float = 60.0
    outer_retries: int = 5
    outer_retry_delay_sec: float = 1.0
    cache_dir: str | Path | None = None
    persist_cache: bool = True


@dataclass(slots=True)
class OpenAICompatibleLlmClient:
    settings: OpenAICompatibleSettings
    transport: HttpTransport | None = None
    _sdk_client: Any = field(default=None, init=False, repr=False)

    def ask_json(self, prompt: str, *, log_title: str, use_cache: bool = True) -> Any:
        return self._ask_with_outer_retry(prompt, resp_type="json", log_title=log_title, use_cache=use_cache)

    def ask_json_validated(self, prompt: str, *, valid_def, log_title: str, use_cache: bool = True) -> Any:
        return self._ask_with_outer_retry(
            prompt,
            resp_type="json",
            log_title=log_title,
            use_cache=use_cache,
            valid_def=valid_def,
        )

    def list_models(self) -> Any:
        if self.transport is not None:
            response = self._get("/models")
            try:
                return decode_response_json(response)
            except Exception as exc:
                raise LlmServiceError(f"LLM models response is not valid JSON: {exc}") from exc
        try:
            models = self._client().models.list(timeout=self.settings.timeout_sec)
        except Exception as exc:
            raise self._sdk_error(exc)
        if hasattr(models, "model_dump"):
            return models.model_dump()
        if hasattr(models, "dict"):
            return models.dict()
        return models

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.settings.user_agent:
            headers["User-Agent"] = self.settings.user_agent
        headers.update(self.settings.extra_headers)
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        return headers

    def _url(self, path: str) -> str:
        return urljoin(normalize_openai_base_url(self.settings.base_url).rstrip("/") + "/", path.lstrip("/"))

    def _client(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        if not self.settings.api_key and self.transport is None:
            raise ValueError("API key is not set")
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LlmServiceError("openai Python package is not available") from exc
        headers = dict(self.settings.extra_headers)
        if self.settings.user_agent:
            headers["User-Agent"] = self.settings.user_agent
        http_client = None
        proxy_url = str(self.settings.proxy_url or "").strip()
        if proxy_url:
            try:
                import httpx

                http_client = httpx.Client(proxy=proxy_url, trust_env=self.settings.trust_env_proxy)
            except Exception as exc:
                raise LlmServiceError(f"LLM proxy client setup failed: {exc}") from exc
        self._sdk_client = OpenAI(
            api_key=self.settings.api_key,
            base_url=normalize_openai_base_url(self.settings.base_url),
            timeout=self.settings.timeout_sec,
            default_headers=headers,
            http_client=http_client,
        )
        return self._sdk_client

    def _ask_with_outer_retry(self, prompt: str, *, resp_type: str, log_title: str, use_cache: bool, valid_def=None) -> Any:
        if not self.settings.api_key and self.transport is None:
            raise ValueError("API key is not set")
        cache_dir = self._cache_dir() if self.settings.persist_cache else None
        if use_cache and cache_dir is not None:
            cached = _load_cache(cache_dir, prompt, resp_type, log_title)
            if cached is not _CACHE_MISS:
                print("use cache response")
                return cached

        last_exc: Exception | None = None
        for attempt in range(self.settings.outer_retries + 1):
            try:
                resp_content = self._create_chat_completion_content(prompt, resp_type=resp_type, log_title=log_title, use_cache=use_cache)
                parsed = parse_json_content(resp_content) if resp_type == "json" else resp_content
                if valid_def:
                    valid_resp = valid_def(parsed)
                    if valid_resp.get("status") != "success":
                        if cache_dir is not None:
                            _save_cache(
                                cache_dir,
                                self.settings.model,
                                prompt,
                                resp_content,
                                resp_type,
                                parsed,
                                log_title="error",
                                message=str(valid_resp.get("message") or ""),
                            )
                        raise ValueError(f"API response error: {valid_resp.get('message')}")
                if cache_dir is not None:
                    _save_cache(cache_dir, self.settings.model, prompt, resp_content, resp_type, parsed, log_title=log_title)
                return parsed
            except LlmRequestError:
                raise
            except Exception as exc:
                last_exc = exc
                print(f"GPT request failed: {exc}, retry: {attempt + 1}/{self.settings.outer_retries}")
                if attempt >= self.settings.outer_retries:
                    break
                time.sleep(self.settings.outer_retry_delay_sec * (2**attempt))
        raise last_exc or LlmServiceError("LLM request failed")

    def _create_chat_completion_content(self, prompt: str, *, resp_type: str, log_title: str, use_cache: bool) -> str:
        payload = build_chat_completion_payload(prompt, self.settings, log_title=log_title, use_cache=use_cache)
        if self.transport is not None:
            response = self._post("/chat/completions", payload)
            return extract_message_content(response)
        response = self._sdk_create_with_retry(payload)
        return extract_message_content(response)

    def _sdk_create_with_retry(self, payload: dict[str, Any]) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                return self._client().chat.completions.create(**payload, timeout=self.settings.timeout_sec)
            except self._transient_openai_errors() as exc:
                last_exc = exc
                if attempt >= self.settings.max_retries:
                    break
                delay = min(self.settings.retry_base_delay_sec * (2**attempt), self.settings.retry_max_delay_sec)
                retry_after = _retry_after_seconds_from_exception(exc)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                delay += random.uniform(0, delay * 0.25)
                print(
                    f"LLM transient error ({type(exc).__name__}): {exc}. "
                    f"Retry {attempt + 1}/{self.settings.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
            except Exception as exc:
                raise self._sdk_error(exc) from exc
        raise self._sdk_error(last_exc) if last_exc else LlmServiceError("LLM request failed")

    def _transient_openai_errors(self):
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        return (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

    def _sdk_error(self, exc: Exception | None) -> LlmAdapterError:
        if exc is None:
            return LlmServiceError("LLM request failed")
        try:
            from openai import APIConnectionError, APITimeoutError, APIStatusError, InternalServerError, RateLimitError
        except Exception:
            return LlmServiceError(f"LLM request failed: {exc}")
        if isinstance(exc, (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)):
            return LlmServiceError(f"LLM connection failure: {exc}")
        if isinstance(exc, APIStatusError):
            status_code = int(getattr(exc, "status_code", 0) or 0)
            if status_code >= 500:
                return LlmServiceError(f"LLM {status_code} server error: {exc}")
            return LlmRequestError(f"LLM {status_code} request error: {exc}")
        return LlmRequestError(f"LLM request failed: {exc}")

    def _cache_dir(self) -> Path:
        if self.settings.cache_dir:
            return Path(self.settings.cache_dir)
        output_dir = os.environ.get("EISTARA_OUTPUT_DIR", "").strip()
        if output_dir:
            return Path(output_dir) / "gpt_log"
        job_dir = os.environ.get("EISTARA_JOB_DIR", "").strip()
        if job_dir:
            return Path(job_dir) / "output" / "gpt_log"
        return Path("output") / "gpt_log"

    def _get(self, path: str) -> HttpResponse:
        try:
            assert self.transport is not None
            response = self.transport.get(self._url(path), headers=self._headers(), timeout=self.settings.timeout_sec)
        except Exception as exc:
            raise LlmServiceError(f"LLM connection failure: {exc}") from exc
        classify_http_response(response)
        return response

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._post_with_retry(path, payload)
        try:
            return decode_response_json(response)
        except Exception as exc:
            raise LlmServiceError(f"LLM response is not valid JSON: {exc}") from exc

    def _post_with_retry(self, path: str, payload: dict[str, Any]) -> HttpResponse:
        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                assert self.transport is not None
                response = self.transport.post(
                    self._url(path),
                    headers=self._headers(),
                    json=payload,
                    timeout=self.settings.timeout_sec,
                )
                classify_http_response(response)
                return response
            except LlmRequestError as exc:
                if not _is_retryable_request_error(exc):
                    raise
                last_error = exc
                if attempt >= self.settings.max_retries:
                    break
                delay = retry_delay_seconds(
                    attempt,
                    response=locals().get("response"),
                    base_delay=self.settings.retry_base_delay_sec,
                    max_delay=self.settings.retry_max_delay_sec,
                )
                print(
                    f"LLM retryable request error ({type(exc).__name__}): {exc}. "
                    f"Retry {attempt + 1}/{self.settings.max_retries} in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
            except Exception as exc:
                service_error = exc if isinstance(exc, LlmServiceError) else LlmServiceError(f"LLM connection failure: {exc}")
                last_error = service_error
                if attempt >= self.settings.max_retries:
                    break
                delay = retry_delay_seconds(
                    attempt,
                    response=locals().get("response"),
                    base_delay=self.settings.retry_base_delay_sec,
                    max_delay=self.settings.retry_max_delay_sec,
                )
                print(
                    f"LLM transient error ({type(service_error).__name__}): {service_error}. "
                    f"Retry {attempt + 1}/{self.settings.max_retries} in {delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
        raise last_error or LlmServiceError("LLM request failed")


def _is_retryable_request_error(exc: LlmRequestError) -> bool:
    text = str(exc).lower()
    return "429" in text or "rate_limit" in text or "rate limit" in text or "concurrency limit" in text


def build_chat_completion_payload(
    prompt: str,
    settings: OpenAICompatibleSettings,
    *,
    log_title: str,
    use_cache: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
        ],
    }
    if settings.temperature is not None:
        payload["temperature"] = settings.temperature
    if settings.response_format_json:
        payload["response_format"] = {"type": "json_object"}
    return payload


def normalize_openai_base_url(base_url: str) -> str:
    text = str(base_url or "").strip()
    if not text:
        return text
    parsed = urlsplit(text if "://" in text else f"https://{text}")
    path = parsed.path.rstrip("/")
    if not _has_version_segment(path):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _has_version_segment(path: str) -> bool:
    return any(part.startswith("v") and part[1:].isdigit() for part in path.strip("/").split("/") if part)


def classify_http_response(response: HttpResponse) -> None:
    if response.status_code >= 500:
        raise LlmServiceError(f"LLM {response.status_code} server error: {_detail(response)}")
    if response.status_code >= 400:
        raise LlmRequestError(f"LLM {response.status_code} request error: {_detail(response)}")
    try:
        response.raise_for_status()
    except Exception as exc:
        raise LlmRequestError(f"LLM request failed: {exc}") from exc


def retry_delay_seconds(
    attempt: int,
    *,
    response: HttpResponse | None = None,
    base_delay: float = 4.0,
    max_delay: float = 60.0,
) -> float:
    delay = min(base_delay * (2**attempt), max_delay)
    retry_after = retry_after_seconds(response)
    if retry_after is not None:
        delay = max(delay, retry_after)
    if delay <= 0:
        return 0.0
    return delay + random.uniform(0, delay * 0.25)


def retry_after_seconds(response: HttpResponse | None) -> float | None:
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    value = None
    try:
        value = headers.get("retry-after") or headers.get("Retry-After")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = parsedate_to_datetime(str(value))
    except (TypeError, ValueError):
        return None
    return max(0.0, parsed.timestamp() - time.time())


def extract_message_content(response_json: Any) -> str:
    if hasattr(response_json, "choices"):
        try:
            return str(response_json.choices[0].message.content)
        except Exception as exc:
            raise LlmServiceError("LLM response is missing choices[0].message.content") from exc
    try:
        return str(response_json["choices"][0]["message"]["content"])
    except Exception as exc:
        raise LlmServiceError("LLM response is missing choices[0].message.content") from exc


def decode_response_json(response: HttpResponse) -> Any:
    content = getattr(response, "content", b"") or b""
    if content:
        return json.loads(content.decode("utf-8-sig"))
    return response.json()


def parse_json_content(content: str) -> Any:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        try:
            import json_repair

            repaired = json_repair.loads(text)
        except Exception:
            preview = text[:300].replace("\n", "\\n")
            raise LlmAdapterError(f"LLM content is not valid JSON: {exc}; preview={preview!r}") from exc
        if repaired in ("", None):
            preview = text[:300].replace("\n", "\\n")
            raise LlmAdapterError(f"LLM content repaired to an empty value; preview={preview!r}") from exc
        return repaired


def _detail(response: HttpResponse) -> str:
    return (getattr(response, "text", "") or getattr(response, "content", b"")[:200].decode("utf-8", errors="ignore"))[:200]


def _retry_after_seconds_from_exception(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cache_path(cache_dir: Path, log_title: str) -> Path:
    return cache_dir / f"{log_title}.json"


def _load_cache(cache_dir: Path, prompt: str, resp_type: str, log_title: str) -> Any:
    with _CACHE_LOCK:
        path = _cache_path(cache_dir, log_title)
        if not path.exists():
            return _CACHE_MISS
        try:
            logs = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return _CACHE_MISS
        if not isinstance(logs, list):
            return _CACHE_MISS
        for item in logs:
            if isinstance(item, dict) and item.get("prompt") == prompt and item.get("resp_type") == resp_type:
                return item.get("resp")
        return _CACHE_MISS


def _save_cache(
    cache_dir: Path,
    model: str,
    prompt: str,
    resp_content: str,
    resp_type: str,
    resp: Any,
    *,
    message: str | None = None,
    log_title: str = "default",
) -> None:
    with _CACHE_LOCK:
        path = _cache_path(cache_dir, log_title)
        path.parent.mkdir(parents=True, exist_ok=True)
        logs = []
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                logs = loaded if isinstance(loaded, list) else []
            except Exception:
                logs = []
        logs.append(
            {
                "model": model,
                "prompt": prompt,
                "resp_content": resp_content,
                "resp_type": resp_type,
                "resp": resp,
                "message": message,
            }
        )
        path.write_text(json.dumps(logs, ensure_ascii=False, indent=4), encoding="utf-8")
