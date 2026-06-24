from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .audio import write_silence_wav
from .models import TtsRequest, TtsSettings


class TtsProviderError(RuntimeError):
    """Base error for content or provider failures."""


class TtsServiceError(TtsProviderError):
    """Infrastructure failure, such as service down, timeout, or 5xx."""


class TtsProvider(Protocol):
    name: str

    def synthesize(self, request: TtsRequest, settings: TtsSettings) -> None:
        """Write audio to request.output_path."""


class ScriptedTtsProvider:
    name = "scripted"

    def __init__(self, failures: list[Exception] | None = None, payload: bytes | None = None):
        self.failures = list(failures or [])
        self.payload = payload
        self.calls: list[TtsRequest] = []

    def synthesize(self, request: TtsRequest, settings: TtsSettings) -> None:
        self.calls.append(request)
        if self.failures:
            raise self.failures.pop(0)
        Path(request.output_path).parent.mkdir(parents=True, exist_ok=True)
        if self.payload is None:
            write_silence_wav(request.output_path)
        else:
            Path(request.output_path).write_bytes(self.payload)
