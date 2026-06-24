from __future__ import annotations

import json as jsonlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.adapters.llm import (
    LlmRequestError,
    LlmServiceError,
    OpenAICompatibleLlmClient,
    OpenAICompatibleSettings,
    build_chat_completion_payload,
    normalize_openai_base_url,
    parse_json_content,
)


@dataclass(slots=True)
class FakeResponse:
    status_code: int = 200
    data: Any = None
    text: str = ""
    content: bytes = b""
    headers: dict[str, str] | None = None

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self.data


class FakeTransport:
    def __init__(self, response: FakeResponse | None = None, error: Exception | None = None, responses: list[FakeResponse] | None = None):
        self.response = response or FakeResponse(data={})
        self.responses = list(responses or [])
        self.error = error
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        if self.error:
            raise self.error
        return self.response

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any], timeout: float) -> FakeResponse:
        self.post_calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        if self.error:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return self.response


def test_build_chat_completion_payload_requests_json() -> None:
    payload = build_chat_completion_payload(
        "translate this",
        OpenAICompatibleSettings(base_url="http://x/v1", model="model-a"),
        log_title="title",
        use_cache=False,
    )

    assert payload["model"] == "model-a"
    assert payload["messages"] == [{"role": "user", "content": "translate this"}]
    assert payload["response_format"] == {"type": "json_object"}
    assert "metadata" not in payload
    assert "temperature" not in payload


def test_openai_compatible_defaults_match_v1_ask_gpt_runtime() -> None:
    settings = OpenAICompatibleSettings(base_url="http://x/v1", model="model-a")

    assert settings.timeout_sec == 300
    assert settings.user_agent == "curl/8.19.0"
    assert settings.trust_env_proxy is True
    assert settings.max_retries == 6
    assert settings.retry_base_delay_sec == 4.0
    assert settings.retry_max_delay_sec == 60.0


def test_openai_compatible_sdk_client_sets_timeout(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("openai.OpenAI", FakeOpenAI)
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://x", model="model-a", api_key="key", timeout_sec=12.5)
    )

    client._client()

    assert captured["timeout"] == 12.5
    assert captured["base_url"] == "http://x/v1"
    assert captured["default_headers"]["User-Agent"] == "curl/8.19.0"


def test_parse_json_content_accepts_fenced_json() -> None:
    assert parse_json_content('```json\n{"ok": true}\n```') == {"ok": True}


def test_parse_json_content_repairs_near_json() -> None:
    assert parse_json_content('{"ok": true,}') == {"ok": True}


def test_openai_compatible_client_ask_json(tmp_path: Path) -> None:
    transport = FakeTransport(
        FakeResponse(
            data={
                "choices": [
                    {
                        "message": {
                            "content": '{"translations":[{"id":1,"text":"hello"}]}',
                        }
                    }
                ]
            }
        )
    )
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://localhost:8000/v1", model="m", api_key="key", cache_dir=tmp_path),
        transport,
    )

    result = client.ask_json("prompt", log_title="translate", use_cache=True)

    assert result["translations"][0]["id"] == 1
    assert transport.post_calls[0]["url"] == "http://localhost:8000/v1/chat/completions"
    assert transport.post_calls[0]["headers"]["Authorization"] == "Bearer key"
    assert transport.post_calls[0]["headers"]["User-Agent"] == "curl/8.19.0"


def test_openai_compatible_client_decodes_transport_json_as_utf8_bytes(tmp_path: Path) -> None:
    correct_content = jsonlib.dumps({"translations": [{"id": 1, "text": "你好"}]}, ensure_ascii=False)
    mojibake_content = jsonlib.dumps({"translations": [{"id": 1, "text": "ä½\xa0å¥½"}]}, ensure_ascii=False)
    response_body = {"choices": [{"message": {"content": correct_content}}]}
    mojibake_body = {"choices": [{"message": {"content": mojibake_content}}]}
    transport = FakeTransport(
        FakeResponse(
            data=mojibake_body,
            content=jsonlib.dumps(response_body, ensure_ascii=False).encode("utf-8"),
        )
    )
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://localhost:8000/v1", model="m", api_key="key", cache_dir=tmp_path),
        transport,
    )

    result = client.ask_json("prompt", log_title="translate", use_cache=True)

    assert result["translations"][0]["text"] == "你好"


def test_openai_compatible_client_can_disable_persistent_cache(tmp_path: Path) -> None:
    transport = FakeTransport(FakeResponse(data={"choices": [{"message": {"content": '{"ok": true}'}}]}))
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(
            base_url="http://localhost:8000/v1",
            model="m",
            api_key="key",
            cache_dir=tmp_path,
            persist_cache=False,
        ),
        transport,
    )

    assert client.ask_json("prompt", log_title="probe", use_cache=False) == {"ok": True}
    assert not (tmp_path / "probe.json").exists()


def test_openai_compatible_client_lists_models() -> None:
    transport = FakeTransport(FakeResponse(data={"data": [{"id": "m"}]}))
    client = OpenAICompatibleLlmClient(OpenAICompatibleSettings(base_url="http://x/v1", model="m"), transport)

    assert client.list_models() == {"data": [{"id": "m"}]}
    assert transport.get_calls[0]["url"] == "http://x/v1/models"


def test_openai_compatible_5xx_is_service_error(tmp_path: Path) -> None:
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://x/v1", model="m", max_retries=0, outer_retries=0, cache_dir=tmp_path),
        FakeTransport(FakeResponse(status_code=500, data={}, text="boom")),
    )

    try:
        client.ask_json("prompt", log_title="x")
    except LlmServiceError as exc:
        assert "500" in str(exc)
    else:
        raise AssertionError("expected LlmServiceError")


def test_openai_compatible_retries_5xx_then_succeeds(tmp_path: Path, capsys) -> None:
    transport = FakeTransport(
        responses=[
            FakeResponse(status_code=502, data={}, text="bad gateway"),
            FakeResponse(data={"choices": [{"message": {"content": '{"ok": true}'}}]}),
        ]
    )
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(
            base_url="http://x/v1",
            model="m",
            max_retries=1,
            outer_retries=0,
            retry_base_delay_sec=0,
            cache_dir=tmp_path,
        ),
        transport,
    )

    assert client.ask_json("prompt", log_title="x") == {"ok": True}
    assert len(transport.post_calls) == 2
    assert "LLM transient error (LlmServiceError)" in capsys.readouterr().out


def test_openai_compatible_4xx_is_request_error(tmp_path: Path) -> None:
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://x/v1", model="m", outer_retries=0, cache_dir=tmp_path),
        FakeTransport(FakeResponse(status_code=401, data={}, text="unauthorized")),
    )

    try:
        client.ask_json("prompt", log_title="x")
    except LlmRequestError as exc:
        assert "401" in str(exc)
    else:
        raise AssertionError("expected LlmRequestError")


def test_openai_compatible_honors_zero_outer_retries_for_4xx(tmp_path: Path) -> None:
    transport = FakeTransport(FakeResponse(status_code=401, data={}, text="unauthorized"))
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://x/v1", model="m", max_retries=3, outer_retries=0, retry_base_delay_sec=0, cache_dir=tmp_path),
        transport,
    )

    try:
        client.ask_json("prompt", log_title="x")
    except LlmRequestError:
        pass
    else:
        raise AssertionError("expected LlmRequestError")
    assert len(transport.post_calls) == 1


def test_openai_compatible_retries_429_request_errors(tmp_path: Path) -> None:
    transport = FakeTransport(
        responses=[
            FakeResponse(status_code=429, data={}, text='{"error":{"message":"Concurrency limit exceeded for account"}}'),
            FakeResponse(data={"choices": [{"message": {"content": "{\"ok\": true}"}}]}),
        ]
    )
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(base_url="http://x/v1", model="m", max_retries=1, outer_retries=0, retry_base_delay_sec=0, cache_dir=tmp_path),
        transport,
    )

    assert client.ask_json("prompt", log_title="x") == {"ok": True}
    assert len(transport.post_calls) == 2


def test_openai_compatible_does_not_outer_retry_request_errors(tmp_path: Path) -> None:
    transport = FakeTransport(FakeResponse(status_code=404, data={}, text="unknown model"))
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(
            base_url="http://x/v1",
            model="m",
            max_retries=3,
            outer_retries=5,
            retry_base_delay_sec=0,
            outer_retry_delay_sec=0,
            cache_dir=tmp_path,
        ),
        transport,
    )

    try:
        client.ask_json("prompt", log_title="x")
    except LlmRequestError:
        pass
    else:
        raise AssertionError("expected LlmRequestError")
    assert len(transport.post_calls) == 1


def test_normalize_openai_base_url_adds_v1_when_no_version_is_present() -> None:
    assert normalize_openai_base_url("https://api.example.com") == "https://api.example.com/v1"
    assert normalize_openai_base_url("https://api.example.com/v1") == "https://api.example.com/v1"
    assert normalize_openai_base_url("https://ark.example.com/anything") == "https://ark.example.com/anything/v1"
    assert normalize_openai_base_url("https://ark.cn-beijing.volces.com/api/v3") == "https://ark.cn-beijing.volces.com/api/v3"
