from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Mapping

from .batching import split_batches
from .llm import LlmClient
from .localization import (
    build_dubbing_length_constraints,
    build_localization_constraints,
    build_localization_prompt,
    build_localization_semantic_repair_prompt,
    build_localization_semantic_review_prompt,
    localization_report,
)
from .models import Terminology, TranslationItem, TranslationResult, TranslationSettings
from .prompting import build_publish_prompt
from .validator import normalize_translation_response, source_latin_allowlist
from eistara.core.timeline import SourceWindow, build_source_windows


class LocalizationBudgetError(ValueError):
    """Raised when second-pass localization does not satisfy spoken-cost budgets."""


@dataclass(slots=True)
class PublishTranslationService:
    llm: LlmClient
    settings: TranslationSettings = TranslationSettings()

    def translate(self, items: list[TranslationItem], terminology: Terminology | None = None) -> TranslationResult:
        terminology = terminology or Terminology()
        translations: dict[int, str] = {}
        warnings: list[str] = []
        localization_reports: list[dict] = []
        batches = split_batches(items, self.settings)
        source_windows = build_source_windows(
            items,
            max_gap_after_sec=self.settings.localization_max_window_gap_sec,
        )
        print(f"Publish fast translation: {len(items)} item(s), {len(batches)} batch(es).")

        draft_translations: dict[int, str] = {}
        for index, batch in enumerate(batches, start=1):
            first_id = batch[0].id
            last_id = batch[-1].id
            print(f"First-pass publish batch {index}/{len(batches)}: id {first_id}-{last_id}")
            result = self.translate_batch_with_fallback(batch, terminology, source_windows=source_windows)
            draft_translations.update(result.translations)
            warnings.extend(result.warnings)

        for index, batch in enumerate(batches, start=1):
            first_id = batch[0].id
            last_id = batch[-1].id
            print(f"Second-pass publish batch {index}/{len(batches)}: id {first_id}-{last_id}")
            batch_drafts = {item.id: draft_translations.get(item.id, "") for item in batch}
            result = self.localize_batch_with_fallback(
                batch,
                terminology,
                batch_drafts,
                source_windows=source_windows,
            )
            translations.update(result.translations)
            warnings.extend(result.warnings)
            localization_reports.extend(result.localization_reports)
        return TranslationResult(translations=translations, warnings=warnings, localization_reports=localization_reports)

    def translate_batch_with_fallback(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
        *,
        source_windows: Mapping[object, SourceWindow] | None = None,
    ) -> TranslationResult:
        try:
            translations = self.translate_batch(batch, terminology, source_windows=source_windows)
            return TranslationResult(translations=translations)
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
                        source_windows=source_windows,
                    )
                    return TranslationResult(translations=translations, warnings=[str(exc)])
                raise
            mid = len(batch) // 2
            print(f"Publish batch failed ({len(batch)} item(s)); splitting into {mid} + {len(batch) - mid}: {exc}")
            left = self.translate_batch_with_fallback(batch[:mid], terminology, source_windows=source_windows)
            right = self.translate_batch_with_fallback(batch[mid:], terminology, source_windows=source_windows)
            merged = dict(left.translations)
            merged.update(right.translations)
            return TranslationResult(translations=merged, warnings=[*left.warnings, *right.warnings, str(exc)])

    def localize_batch_with_fallback(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
        translations: dict[int, str],
        *,
        source_windows: Mapping[object, SourceWindow] | None = None,
    ) -> TranslationResult:
        try:
            return self.localize_batch(batch, terminology, translations, source_windows=source_windows)
        except Exception as exc:
            if _is_non_splittable_llm_error(exc):
                raise
            if len(batch) <= 1:
                if "untranslated English remains" in str(exc):
                    localized = self.localize_batch(
                        batch,
                        terminology,
                        translations,
                        enforce_latin=False,
                        use_cache=False,
                        source_windows=source_windows,
                    )
                    return TranslationResult(
                        translations=localized.translations,
                        warnings=[str(exc), *localized.warnings],
                        localization_reports=localized.localization_reports,
                    )
                raise
            mid = len(batch) // 2
            print(f"Localization batch failed ({len(batch)} item(s)); splitting into {mid} + {len(batch) - mid}: {exc}")
            left_batch = batch[:mid]
            right_batch = batch[mid:]
            left_translations = {item.id: translations.get(item.id, "") for item in left_batch}
            right_translations = {item.id: translations.get(item.id, "") for item in right_batch}
            left = self.localize_batch_with_fallback(
                left_batch,
                terminology,
                left_translations,
                source_windows=source_windows,
            )
            right = self.localize_batch_with_fallback(
                right_batch,
                terminology,
                right_translations,
                source_windows=source_windows,
            )
            merged = dict(left.translations)
            merged.update(right.translations)
            return TranslationResult(
                translations=merged,
                warnings=[*left.warnings, *right.warnings, str(exc)],
                localization_reports=[*left.localization_reports, *right.localization_reports],
            )

    def localize_batch(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
        translations: dict[int, str],
        *,
        enforce_latin: bool | None = None,
        use_cache: bool = True,
        source_windows: Mapping[object, SourceWindow] | None = None,
    ) -> TranslationResult:
        constraints = build_localization_constraints(batch, translations, self.settings, source_windows=source_windows)
        enforce = self.settings.enforce_latin if enforce_latin is None else enforce_latin
        expected_ids = [item.id for item in batch]
        source_latin_by_id = {item.id: source_latin_allowlist(item.source) for item in batch}
        prompt = build_localization_prompt(batch, terminology, self.settings, constraints)

        def valid_response(response_data):
            try:
                normalized = normalize_translation_response(
                    response_data,
                    expected_ids,
                    source_latin_by_id,
                    target_language=self.settings.target_language,
                    enforce_latin=enforce,
                )
            except Exception as exc:
                return {"status": "error", "message": str(exc)}
            budget_error = _localization_budget_error(localization_report(constraints, normalized))
            if budget_error:
                return {"status": "error", "message": budget_error}
            return {"status": "success", "message": "ok"}

        ask_json_validated = getattr(self.llm, "ask_json_validated", None)
        if ask_json_validated:
            last_exc: Exception | None = None
            disable_cache = False
            for attempt in range(6):
                try:
                    response = ask_json_validated(
                        prompt,
                        valid_def=valid_response,
                        log_title="translate_publish_localize",
                        use_cache=use_cache and not disable_cache,
                    )
                    localized = normalize_translation_response(
                        response,
                        expected_ids,
                        source_latin_by_id,
                        target_language=self.settings.target_language,
                        enforce_latin=enforce,
                    )
                    return self._finalize_localization(
                        batch,
                        terminology,
                        constraints,
                        localized,
                        expected_ids,
                        source_latin_by_id,
                        enforce=enforce,
                        use_cache=use_cache and not disable_cache,
                    )
                except Exception as exc:
                    last_exc = exc
                    if _is_non_splittable_llm_error(exc):
                        raise
                    disable_cache = True
                    print(f"Localization validation failed: {exc}, retry: {attempt + 1}/5")
                    if attempt >= 5:
                        break
                    time.sleep(1 * (2**attempt))
            raise last_exc or RuntimeError("Localization request failed")

        last_exc: Exception | None = None
        disable_cache = False
        for attempt in range(6):
            try:
                response = self.llm.ask_json(
                    prompt,
                    log_title="translate_publish_localize",
                    use_cache=use_cache and not disable_cache,
                )
                localized = normalize_translation_response(
                    response,
                    expected_ids,
                    source_latin_by_id,
                    target_language=self.settings.target_language,
                    enforce_latin=enforce,
                )
                _raise_for_localization_budget(localization_report(constraints, localized))
                return self._finalize_localization(
                    batch,
                    terminology,
                    constraints,
                    localized,
                    expected_ids,
                    source_latin_by_id,
                    enforce=enforce,
                    use_cache=use_cache and not disable_cache,
                )
            except Exception as exc:
                last_exc = exc
                if _is_non_splittable_llm_error(exc):
                    raise
                if isinstance(exc, LocalizationBudgetError):
                    disable_cache = True
                print(f"Localization request failed: {exc}, retry: {attempt + 1}/5")
                if attempt >= 5:
                    break
                time.sleep(1 * (2**attempt))
        raise last_exc or RuntimeError("Localization request failed")

    def translate_batch(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
        *,
        enforce_latin: bool | None = None,
        use_cache: bool = True,
        source_windows: Mapping[object, SourceWindow] | None = None,
    ) -> dict[int, str]:
        enforce = self.settings.enforce_latin if enforce_latin is None else enforce_latin
        expected_ids = [item.id for item in batch]
        source_latin_by_id = {item.id: source_latin_allowlist(item.source) for item in batch}
        length_constraints = build_dubbing_length_constraints(batch, self.settings, source_windows=source_windows)
        prompt = build_publish_prompt(batch, terminology, self.settings, length_constraints)

        def valid_response(response_data):
            try:
                normalized = normalize_translation_response(
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
            normalized = normalize_translation_response(
                response,
                expected_ids,
                source_latin_by_id,
                target_language=self.settings.target_language,
                enforce_latin=enforce,
            )
            return normalized

        last_exc: Exception | None = None
        for attempt in range(6):
            try:
                response = self.llm.ask_json(prompt, log_title="translate_publish_fast", use_cache=use_cache)
                normalized = normalize_translation_response(
                    response,
                    expected_ids,
                    source_latin_by_id,
                    target_language=self.settings.target_language,
                    enforce_latin=enforce,
                )
                return normalized
            except Exception as exc:
                last_exc = exc
                if _is_non_splittable_llm_error(exc):
                    raise
                print(f"GPT request failed: {exc}, retry: {attempt + 1}/5")
                if attempt >= 5:
                    break
                time.sleep(1 * (2**attempt))
        raise last_exc or RuntimeError("Translation request failed")

    def _finalize_localization(
        self,
        batch: list[TranslationItem],
        terminology: Terminology,
        constraints: list,
        localized: dict[int, str],
        expected_ids: list[int],
        source_latin_by_id: dict[int, set[str]],
        *,
        enforce: bool,
        use_cache: bool,
    ) -> TranslationResult:
        warnings: list[str] = []
        report = localization_report(constraints, localized)
        try:
            review_prompt, candidates = build_localization_semantic_review_prompt(
                batch,
                terminology,
                self.settings,
                constraints,
                localized,
            )
            semantic_report = {
                "enabled": True,
                "candidate_count": len(candidates),
                "issue_count": 0,
                "repaired_count": 0,
                "issues": [],
            }
            if candidates:
                review_response = self.llm.ask_json(
                    review_prompt,
                    log_title="translate_publish_localize_review",
                    use_cache=use_cache,
                )
                issues = _normalize_semantic_issues(review_response, expected_ids)
                semantic_report["issues"] = issues
                semantic_report["issue_count"] = len(issues)
                repair_issues = [
                    issue
                    for issue in issues
                    if str(issue.get("severity", "")).lower() in {"major", "critical"}
                ]
                if repair_issues:
                    repair_prompt = build_localization_semantic_repair_prompt(
                        batch,
                        terminology,
                        self.settings,
                        constraints,
                        localized,
                        repair_issues,
                    )
                    repair_response = self.llm.ask_json(
                        repair_prompt,
                        log_title="translate_publish_localize_repair",
                        use_cache=False,
                    )
                    repaired_ids = [int(issue["id"]) for issue in repair_issues]
                    repaired = normalize_translation_response(
                        repair_response,
                        repaired_ids,
                        source_latin_by_id,
                        target_language=self.settings.target_language,
                        enforce_latin=enforce,
                    )
                    localized = {**localized, **repaired}
                    semantic_report["repaired_count"] = len(repaired)
                    semantic_report["repaired_ids"] = sorted(repaired)
                    report = localization_report(constraints, localized)
            report["semantic_review"] = semantic_report
        except Exception as exc:
            raise RuntimeError(f"Localization semantic review failed: {exc}") from exc
        _raise_for_localization_budget(report)
        warnings.extend(_localization_budget_warnings(report))
        warnings.extend(_localization_semantic_warnings(report))
        return TranslationResult(translations=localized, warnings=warnings, localization_reports=[report])


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


def _localization_budget_warnings(report: dict) -> list[str]:
    summary = dict(report.get("summary") or {})
    final_count = int(summary.get("final_over_limit_count") or 0)
    under_min_count = int(summary.get("final_under_min_count") or 0)
    if final_count <= 0 and under_min_count <= 0:
        return []
    return [
        "Dubbing spoken-cost budget failed "
        f"(final={final_count}, under_min={under_min_count})."
    ]


def _localization_budget_error(report: dict) -> str | None:
    violations = _localization_budget_violations(report)
    if not violations:
        return None
    preview = ", ".join(
        f"id={item['id']} cost={item.get('final_spoken_cost')} range={item.get('min_spoken_cost')}-{item.get('max_spoken_cost')}"
        for item in violations[:8]
    )
    extra = "" if len(violations) <= 8 else f", +{len(violations) - 8} more"
    return f"Localization spoken-cost budget violation(s): {preview}{extra}"


def _raise_for_localization_budget(report: dict) -> None:
    message = _localization_budget_error(report)
    if message:
        raise LocalizationBudgetError(message)


def _localization_budget_violations(report: dict) -> list[dict]:
    violations: list[dict] = []
    for item in report.get("items") or []:
        if not isinstance(item, dict):
            continue
        if bool(item.get("final_over_limit")) or bool(item.get("final_under_min")):
            violations.append(item)
    return violations


def _normalize_semantic_issues(response_data, expected_ids: list[int]) -> list[dict]:
    if not isinstance(response_data, dict):
        raise ValueError("Semantic review response must be an object")
    raw_issues = response_data.get("issues", [])
    if raw_issues is None:
        return []
    if not isinstance(raw_issues, list):
        raise ValueError("Semantic review issues must be a list")
    expected = set(expected_ids)
    issues: list[dict] = []
    for raw in raw_issues:
        if not isinstance(raw, dict):
            continue
        try:
            item_id = int(raw.get("id"))
        except (TypeError, ValueError):
            continue
        if item_id not in expected:
            continue
        severity = str(raw.get("severity") or "major").strip().lower()
        if severity not in {"minor", "major", "critical"}:
            severity = "major"
        issues.append(
            {
                "id": item_id,
                "severity": severity,
                "issue_type": str(raw.get("issue_type") or "semantic").strip() or "semantic",
                "missing_meaning": str(raw.get("missing_meaning") or "").strip(),
                "repair_instruction": str(raw.get("repair_instruction") or "").strip(),
            }
        )
    return issues


def _localization_semantic_warnings(report: dict) -> list[str]:
    semantic = dict(report.get("semantic_review") or {})
    if semantic.get("failed"):
        return []
    issue_count = int(semantic.get("issue_count") or 0)
    repaired_count = int(semantic.get("repaired_count") or 0)
    if issue_count <= 0:
        return []
    return [
        "Localization semantic review found "
        f"{issue_count} issue(s), repaired {repaired_count}; debug build records this in the localization report."
    ]
