from __future__ import annotations

import time
from dataclasses import dataclass

from .batching import split_batches
from .llm import LlmClient
from .models import Terminology, TranslationItem, TranslationResult, TranslationSettings
from .pacing import build_pacing_budget
from .prompting import build_publish_prompt
from .validator import normalize_translation_response, source_latin_allowlist


@dataclass(slots=True)
class PublishTranslationService:
    llm: LlmClient
    settings: TranslationSettings = TranslationSettings()

    def translate(self, items: list[TranslationItem], terminology: Terminology | None = None) -> TranslationResult:
        terminology = terminology or Terminology()
        translations: dict[int, str] = {}
        warnings: list[str] = []
        batches = split_batches(items, self.settings)
        print(f"Publish fast translation: {len(items)} item(s), {len(batches)} batch(es).")
        for index, batch in enumerate(batches, start=1):
            first_id = batch[0].id
            last_id = batch[-1].id
            print(f"Translating publish batch {index}/{len(batches)}: id {first_id}-{last_id}")
            result = self.translate_batch_with_fallback(batch, terminology)
            translations.update(result.translations)
            warnings.extend(result.warnings)
        return TranslationResult(translations=translations, warnings=warnings)

    def translate_batch_with_fallback(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
    ) -> TranslationResult:
        try:
            return TranslationResult(translations=self.translate_batch(batch, terminology))
        except Exception as exc:
            if _is_non_splittable_llm_error(exc):
                raise
            if len(batch) <= 1:
                if "untranslated English remains" in str(exc):
                    translations = self.translate_batch(
                        batch,
                        terminology,
                        enforce_latin=False,
                        use_cache=False,
                    )
                    return TranslationResult(translations=translations, warnings=[str(exc)])
                raise
            mid = len(batch) // 2
            print(f"Publish batch failed ({len(batch)} item(s)); splitting into {mid} + {len(batch) - mid}: {exc}")
            left = self.translate_batch_with_fallback(batch[:mid], terminology)
            right = self.translate_batch_with_fallback(batch[mid:], terminology)
            merged = dict(left.translations)
            merged.update(right.translations)
            return TranslationResult(translations=merged, warnings=[*left.warnings, *right.warnings, str(exc)])

    def translate_batch(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
        *,
        enforce_latin: bool | None = None,
        use_cache: bool = True,
    ) -> dict[int, str]:
        enforce = self.settings.enforce_latin if enforce_latin is None else enforce_latin
        expected_ids = [item.id for item in batch]
        source_latin_by_id = {item.id: source_latin_allowlist(item.source) for item in batch}
        pacing_budget = build_pacing_budget(batch, self.settings)
        prompt = build_publish_prompt(batch, terminology, self.settings, pacing_budget)

        def valid_response(response_data):
            try:
                normalize_translation_response(
                    response_data,
                    expected_ids,
                    source_latin_by_id,
                    target_language=self.settings.target_language,
                    enforce_latin=enforce,
                )
            except Exception as exc:
                return {"status": "error", "message": str(exc)}
            return {"status": "success", "message": "ok"}

        ask_json_validated = getattr(self.llm, "ask_json_validated", None)
        if ask_json_validated:
            response = ask_json_validated(
                prompt,
                valid_def=valid_response,
                log_title="translate_publish_fast",
                use_cache=use_cache,
            )
            return normalize_translation_response(
                response,
                expected_ids,
                source_latin_by_id,
                target_language=self.settings.target_language,
                enforce_latin=enforce,
            )

        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                response = self.llm.ask_json(prompt, log_title="translate_publish_fast", use_cache=use_cache)
                return normalize_translation_response(
                    response,
                    expected_ids,
                    source_latin_by_id,
                    target_language=self.settings.target_language,
                    enforce_latin=enforce,
                )
            except Exception as exc:
                last_exc = exc
                if _is_non_splittable_llm_error(exc):
                    raise
                print(f"GPT request failed: {exc}, retry: {attempt + 1}/5")
                if attempt >= 5:
                    break
                time.sleep(1 * (2**attempt))
        raise last_exc or RuntimeError("Translation request failed")


def _is_non_splittable_llm_error(exc: Exception) -> bool:
    text = str(exc).lower()
    class_name = exc.__class__.__name__.lower()
    return (
        "llmrequesterror" in class_name
        or "llmserviceerror" in class_name
        or "429" in text
        or "524" in text
        or "rate_limit" in text
        or "rate limit" in text
        or "concurrency limit" in text
        or "read timed out" in text
        or "connection failure" in text
    )
