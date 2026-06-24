from __future__ import annotations

import json
from typing import Iterable

from .models import Terminology, TranslationItem, TranslationPacingBudget, TranslationSettings
from .pacing import count_english_words


def format_terms(terms: Iterable[dict], limit: int = 80) -> str:
    lines: list[str] = []
    for item in list(terms)[:limit]:
        src = str(item.get("src", "")).strip()
        tgt = str(item.get("tgt", "")).strip()
        note = str(item.get("note", "")).strip()
        if not src:
            continue
        lines.append(f"- {src} => {tgt} ({note})" if note else f"- {src} => {tgt}")
    return "\n".join(lines) if lines else "None"


def build_publish_prompt(
    batch: list[TranslationItem],
    terminology: Terminology,
    settings: TranslationSettings,
    pacing_budget: TranslationPacingBudget | None = None,
) -> str:
    theme = terminology.theme.strip() if settings.use_summary else ""
    terms_text = format_terms(terminology.terms)
    batch_json = json.dumps([_prompt_item(item, pacing_budget) for item in batch], ensure_ascii=False, indent=2)
    pacing_enabled = bool(pacing_budget and pacing_budget.enabled)
    pacing_text = _format_pacing_budget(pacing_budget) if pacing_enabled else ""
    timing_requirement = (
        "5. Use the start/end/duration as pacing context for spoken dubbing.\n"
        "   Do not mechanically fit every line to the source duration, but respect the batch pacing budget when one is provided."
        if pacing_enabled
        else "5. Use the start/end/duration only as context for scene rhythm and speech flow.\n"
        "   Do not compress, omit, or summarize meaning just to fit the original timing."
    )
    pacing_requirement = (
        "10. When a dubbing pacing budget is enabled, treat it as an independent lock for this batch only.\n"
        "   Do not make this batch overly short or overly long to compensate for other batches.\n"
        "   Stay inside this batch's Chinese character range while preserving all facts and natural spoken Chinese.\n"
        "   Treat item-level suggested Chinese character counts as flexible guidance; the batch total is more important.\n"
        "11. Return only valid JSON with this schema:\n"
        '   {"translations":[{"id":1,"text":"translated text"}]}'
        if pacing_enabled
        else "10. Translate as complete, natural Chinese narration for dubbing.\n"
        "   Do not summarize, omit, or compress meaning to chase timing.\n"
        "   Keep the spoken rhythm clear and human, but let meaning and natural phrasing lead.\n"
        "11. Return only valid JSON with this schema:\n"
        '   {"translations":[{"id":1,"text":"translated text"}]}'
    )
    pacing_section = f"\n\nDubbing pacing budget:\n{pacing_text}" if pacing_text else ""

    return f"""
You are a professional video dubbing translator.

Translate the subtitle blocks from {settings.source_language} into {settings.target_language}.
This project is publish-first dubbing. Translate for natural spoken delivery and complete meaning.

Requirements:
1. Preserve the exact number of items and the exact id values.
2. Translate every source item completely. Do not omit, summarize, or add new facts.
3. Make the translation natural, spoken, and suitable for Chinese dubbing.
4. Keep terminology, names, numbers, and references accurate and consistent.
{timing_requirement}
6. If the source quotes an English example sentence, translate the meaning into {settings.target_language};
   do not leave long English clauses in the dubbing text.
7. Keep only true proper nouns, brand names, acronyms, or product names in Latin letters
   when they are normally spoken that way, such as ChatGPT, AI, YouTube, GPT, or API.
8. The output text is the TTS script first. Subtitles will be generated later from the TTS timeline.
   Do not add line breaks, subtitle markup, numbering, or timing text.
9. For readability and later one-line subtitle splitting, write in short spoken clauses.
   Prefer natural Chinese punctuation every 8-20 Chinese characters when the sentence is long.
   Do not force each item to be short; keep the meaning complete for dubbing.
{pacing_requirement}

Video/theme context:
{theme or "None"}

Terminology:
{terms_text}{pacing_section}

Input subtitle blocks:
{batch_json}
""".strip()


def _prompt_item(item: TranslationItem, pacing_budget: TranslationPacingBudget | None) -> dict:
    data = item.to_prompt_dict()
    if pacing_budget and pacing_budget.enabled and pacing_budget.english_words > 0:
        item_words = count_english_words(item.source)
        share = item_words / pacing_budget.english_words if item_words > 0 else 0.0
        data["source_english_words"] = item_words
        data["suggested_zh_chars"] = max(1, round(pacing_budget.target_zh_chars * share)) if share > 0 else 0
    return data


def _format_pacing_budget(pacing_budget: TranslationPacingBudget | None) -> str:
    if pacing_budget is None:
        return "None"
    data = pacing_budget.to_prompt_dict()
    if not pacing_budget.enabled:
        guidance = "No fixed Chinese character count is applied; prioritize complete, natural spoken Chinese."
        if data["reason"] == "watch_pressure_avoid_unnecessary_expansion":
            guidance = (
                "No fixed Chinese character count is applied. Keep the translation natural and complete, "
                "but avoid unnecessary explanatory expansion."
            )
        return (
            "Disabled for this batch "
            f"(level={data['level']}, predicted_pressure={data['predicted_pressure']}, reason={data['reason']}). "
            f"{guidance}"
        )
    return "\n".join(
        [
            f"- enabled: true",
            f"- level: {data['level']}",
            f"- predicted_pressure: {data['predicted_pressure']}",
            f"- pressure_at_minimum_zh_ratio: {data['pressure_at_min_zh_ratio']}",
            f"- target_pressure: {data['target_pressure']}",
            f"- soft_pressure: {data['soft_pressure']}",
            f"- batch_source_duration_sec: {data['source_duration_sec']}",
            f"- batch_english_words: {data['english_words']}",
            f"- estimated_tts_chars_per_sec: {data['estimated_tts_chars_per_sec']}",
            f"- minimum_zh_chars_per_en_word: {data['min_zh_chars_per_en_word']}",
            f"- natural_zh_chars_per_en_word: {data['natural_min_zh_chars_per_en_word']}-{data['natural_max_zh_chars_per_en_word']}",
            f"- target_zh_chars_per_en_word: {data['target_zh_chars_per_en_word']}",
            f"- max_zh_chars_per_en_word: {data['max_zh_chars_per_en_word']}",
            f"- minimum_total_chinese_chars: {data['min_zh_chars']}",
            f"- target_total_chinese_chars: {data['target_zh_chars']}",
            f"- max_total_chinese_chars: {data['hard_zh_chars']}",
            "- instruction: this is a per-batch lock. Keep this batch's total Chinese Han-character count "
            "between minimum_total_chinese_chars and max_total_chinese_chars, aiming near target_total_chinese_chars. "
            "Count only Chinese Han characters for this range; do not count punctuation, spaces, digits, or Latin names. "
            "The minimum is a quality floor, not a compression target; the max is the pacing upper limit, not a request "
            "to make the translation as short as possible. "
            "Do not intentionally compress or pad this batch to balance other batches. "
            "If pressure_at_minimum_zh_ratio is already above soft_pressure, do not cut below "
            "minimum_total_chinese_chars; preserve meaning and accept the pacing risk.",
        ]
    )
