from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .audio import has_positive_audio_duration, wav_duration_sec, write_silence_wav
from .cache import TtsCachePolicy
from .models import TtsRequest, TtsResult, TtsSettings
from .postprocess import analyze_generated_tts_audio, postprocess_generated_tts_audio_with_report
from .providers import TtsProvider, TtsProviderError, TtsServiceError
from .text import clean_text_for_tts, is_silent_tts_text


@dataclass(slots=True)
class TtsService:
    provider: TtsProvider
    settings: TtsSettings = TtsSettings()
    sleep: Callable[[float], None] = time.sleep
    text_corrector: Callable[[str], str] | None = None
    cache: TtsCachePolicy = field(init=False)

    def __post_init__(self) -> None:
        self.cache = TtsCachePolicy(self.settings)

    def synthesize(self, request: TtsRequest) -> TtsResult:
        cleaned_text = clean_text_for_tts(request.text)
        normalized_request = TtsRequest(
            text=cleaned_text,
            output_path=request.output_path,
            segment_id=request.segment_id,
            voice=request.voice,
            speaker=request.speaker,
            metadata=request.metadata,
        )
        metadata = self.cache.build_metadata(normalized_request, cleaned_text)
        if self.cache.should_skip(normalized_request.output_path, metadata):
            quality = analyze_generated_tts_audio(normalized_request.output_path, self.settings.audio_config)
            return TtsResult(
                output_path=normalized_request.output_path,
                cached=True,
                warnings=quality.warnings,
                metadata={"audio_quality": quality.to_dict()},
            )

        if is_silent_tts_text(cleaned_text):
            self._write_placeholder_audio(normalized_request.output_path)
            self.cache.write_metadata(normalized_request.output_path, metadata)
            return TtsResult(output_path=normalized_request.output_path, duration_sec=0.1, warnings=["silent placeholder"])

        last_error: Exception | None = None
        last_was_service_error = False
        current_request = normalized_request
        for attempt in range(self.settings.max_retries):
            try:
                if (
                    attempt >= self.settings.max_retries - 1
                    and not last_was_service_error
                    and self.text_corrector is not None
                ):
                    corrected_text = clean_text_for_tts(self.text_corrector(current_request.text))
                    current_request = TtsRequest(
                        text=corrected_text or current_request.text,
                        output_path=current_request.output_path,
                        segment_id=current_request.segment_id,
                        voice=current_request.voice,
                        speaker=current_request.speaker,
                        metadata=current_request.metadata,
                    )
                self.provider.synthesize(current_request, self.settings)
                quality = postprocess_generated_tts_audio_with_report(current_request.output_path, self.settings.audio_config)
                warnings = quality.warnings
                if not has_positive_audio_duration(current_request.output_path):
                    self.cache.remove(current_request.output_path)
                    if attempt >= self.settings.max_retries - 1:
                        self._write_placeholder_audio(current_request.output_path)
                        self.cache.write_metadata(current_request.output_path, metadata)
                        return TtsResult(
                            output_path=current_request.output_path,
                            duration_sec=0.1,
                            warnings=["zero-duration placeholder"],
                        )
                    raise TtsProviderError("TTS provider did not write positive-duration audio")
                self.cache.write_metadata(current_request.output_path, metadata)
                return TtsResult(
                    output_path=current_request.output_path,
                    duration_sec=wav_duration_sec(current_request.output_path),
                    warnings=warnings,
                    metadata={"audio_quality": quality.to_dict()},
                )
            except TtsServiceError as exc:
                last_error = exc
                last_was_service_error = True
                if attempt >= self.settings.max_retries - 1:
                    break
                delay = self.settings.service_backoff_base_sec * (2**attempt)
                delay += random.uniform(0, delay * 0.25)
                self.sleep(delay)
                self._probe_provider_ready()
            except Exception as exc:
                last_error = exc
                last_was_service_error = False
                if attempt >= self.settings.max_retries - 1:
                    break
                self.sleep(0)
        self.cache.remove(normalized_request.output_path)
        raise TtsServiceError(f"TTS failed after {self.settings.max_retries} attempts: {last_error}") from last_error

    def _write_placeholder_audio(self, output_path: Path) -> None:
        write_silence_wav(output_path)

    def _probe_provider_ready(self) -> None:
        check_ready = getattr(self.provider, "check_ready", None)
        if check_ready is None:
            return
        try:
            check_ready(self.settings)
        except TtsServiceError:
            return
        except Exception:
            return
