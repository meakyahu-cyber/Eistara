from __future__ import annotations

from typing import Any, Protocol


class LlmClient(Protocol):
    def ask_json(self, prompt: str, *, log_title: str, use_cache: bool = True) -> Any:
        """Return a parsed JSON-like object from an LLM."""


class ScriptedLlmClient:
    """Tiny test client that returns preloaded responses in order."""

    def __init__(self, responses: list[Any]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def ask_json(self, prompt: str, *, log_title: str, use_cache: bool = True) -> Any:
        self.calls.append({"prompt": prompt, "log_title": log_title, "use_cache": use_cache})
        if not self.responses:
            raise RuntimeError("No scripted LLM response left")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response
