from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .llm import LlmClient
from .models import Terminology, TranslationItem, TranslationSettings


@dataclass(frozen=True, slots=True)
class TerminologySummaryResult:
    path: Path
    terminology: Terminology
    custom_terms_count: int = 0


def build_summary_prompt(
    source_content: str,
    custom_terms_json: dict[str, Any] | None,
    settings: TranslationSettings,
) -> str:
    source_language = settings.source_language
    target_language = settings.target_language

    terms_note = ""
    if custom_terms_json:
        terms_list = []
        for term in custom_terms_json["terms"]:
            terms_list.append(f"- {term['src']}: {term['tgt']} ({term['note']})")
        terms_note = "\n### Existing Terms\nPlease exclude these terms in your extraction:\n" + "\n".join(terms_list)

    return f"""
## Role
You are a video translation expert and terminology consultant, specializing in {source_language} comprehension and {target_language} expression optimization.

## Task
For the provided {source_language} video text:
1. Summarize main topic in two sentences
2. Extract professional terms/names with {target_language} translations (excluding existing terms)
3. Provide brief explanation for each term

{terms_note}

Steps:
1. Topic Summary:
   - Quick scan for general understanding
   - Write two sentences: first for main topic, second for key point
2. Term Extraction:
   - Mark professional terms and names (excluding those listed in Existing Terms)
   - Provide {target_language} translation or keep original
   - Add brief explanation
   - Extract less than 15 terms

## INPUT
<text>
{source_content}
</text>

## Output in only JSON format and no other text
{{
  "theme": "Two-sentence video summary",
  "terms": [
    {{
      "src": "{source_language} term",
      "tgt": "{target_language} translation or original",
      "note": "Brief explanation"
    }}
  ]
}}
""".strip()


def generate_terminology_summary(
    llm: LlmClient,
    items: list[TranslationItem],
    settings: TranslationSettings,
    output_dir: Path,
    *,
    custom_terms_path: Path | None = None,
) -> TerminologySummaryResult:
    source_content = combine_source_content(items, settings.summary_length)
    custom_terms = load_custom_terms(custom_terms_path)
    custom_terms_json = {"terms": custom_terms} if custom_terms_path is not None or custom_terms else None
    prompt = build_summary_prompt(source_content, custom_terms_json, settings)

    ask_json_validated = getattr(llm, "ask_json_validated", None)
    if ask_json_validated:
        response = ask_json_validated(
            prompt,
            valid_def=validate_summary_response,
            log_title="summary",
        )
    else:
        response = llm.ask_json(prompt, log_title="summary")
        validation = validate_summary_response(response)
        if validation.get("status") != "success":
            raise ValueError(str(validation.get("message") or "Invalid summary response"))

    summary = normalize_summary_response(response)
    summary["terms"].extend(custom_terms)

    path = output_dir / "log" / "terminology.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
    return TerminologySummaryResult(
        path=path,
        terminology=Terminology(theme=str(summary.get("theme") or ""), terms=tuple(summary.get("terms") or ())),
        custom_terms_count=len(custom_terms),
    )


def combine_source_content(items: list[TranslationItem], summary_length: int) -> str:
    cleaned_sentences = [str(item.source).strip() for item in items]
    combined_text = " ".join(cleaned_sentences)
    return combined_text[: max(0, int(summary_length))]


def load_custom_terms(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    custom_terms = pd.read_excel(path)
    terms: list[dict[str, str]] = []
    for _, row in custom_terms.iterrows():
        terms.append(
            {
                "src": str(row.iloc[0]) if len(row) > 0 else "",
                "tgt": str(row.iloc[1]) if len(row) > 1 else "",
                "note": str(row.iloc[2]) if len(row) > 2 else "",
            }
        )
    return terms


def validate_summary_response(response_data: Any) -> dict[str, str]:
    required_keys = {"src", "tgt", "note"}
    if not isinstance(response_data, dict) or "terms" not in response_data:
        return {"status": "error", "message": "Invalid response format"}
    if not isinstance(response_data["terms"], list):
        return {"status": "error", "message": "Invalid response format"}
    for term in response_data["terms"]:
        if not isinstance(term, dict) or not all(key in term for key in required_keys):
            return {"status": "error", "message": "Invalid response format"}
    return {"status": "success", "message": "Summary completed"}


def normalize_summary_response(response_data: Any) -> dict[str, Any]:
    validation = validate_summary_response(response_data)
    if validation.get("status") != "success":
        raise ValueError(str(validation.get("message") or "Invalid summary response"))
    assert isinstance(response_data, dict)
    return {
        "theme": str(response_data.get("theme") or ""),
        "terms": [
            {
                "src": str(term.get("src") or ""),
                "tgt": str(term.get("tgt") or ""),
                "note": str(term.get("note") or ""),
            }
            for term in response_data.get("terms") or []
        ],
    }
