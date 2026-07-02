from __future__ import annotations

import json
from typing import Any, Iterable

from .models import Terminology, TranslationItem, TranslationSettings


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
    length_constraints: list[Any] | None = None,
) -> str:
    theme = terminology.theme.strip() if settings.use_summary else ""
    terms_text = format_terms(terminology.terms)
    constraint_by_id = {constraint.id: constraint for constraint in length_constraints or []}
    has_spoken_cost_constraints = any(
        getattr(constraint, "max_spoken_cost", 0) > 0 for constraint in constraint_by_id.values()
    )
    batch_json = json.dumps(
        [_prompt_item(item, constraint_by_id.get(item.id)) for item in batch],
        ensure_ascii=False,
        indent=2,
    )
    timing_requirement = (
        "5. Use the start/end/duration and dubbing_window as hard timing context for spoken dubbing.\n"
        "   For each item with hard_limit_applies=true, write the shortest accurate spoken Chinese that fits max_spoken_cost.\n"
        "   Preserve the core facts, numbers, names, causal links, and conclusions; compress filler and decorative wording first."
        if has_spoken_cost_constraints
        else "5. Use the start/end/duration only as context for scene rhythm and speech flow.\n"
        "   Do not compress, omit, or summarize meaning just to fit the original timing."
    )
    narration_requirement = (
        "10. Translate as complete, natural Chinese narration for dubbing.\n"
        "   Keep the spoken rhythm clear and human while respecting each hard timing budget.\n"
        "11. Return only valid JSON with this schema:\n"
        '   {"translations":[{"id":1,"text":"translated text"}]}'
    )
    length_requirement = (
        "\n12. Dubbing window hard rule:\n"
        "   Each input item may include dubbing_window with target_spoken_cost and max_spoken_cost.\n"
        "   max_spoken_cost is a hard upper bound for every item with hard_limit_applies=true.\n"
        "   Spoken cost includes Han characters, Arabic digits, years, percentages, Latin names/acronyms, and punctuation pauses.\n"
        "   Aim near target_spoken_cost, and do not exceed max_spoken_cost.\n"
        "   Preserve core facts, causal links, numbers, names, and conclusions; compress filler, repetition, and decorative modifiers first.\n"
        "   Use concise spoken forms for numbers and names when accurate, but do not falsify exact values."
        if has_spoken_cost_constraints
        else ""
    )

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
{narration_requirement}
{length_requirement}

Video/theme context:
{theme or "None"}

Terminology:
{terms_text}

Input subtitle blocks:
{batch_json}
""".strip()


def _prompt_item(
    item: TranslationItem,
    length_constraint: Any | None,
) -> dict:
    data = item.to_prompt_dict()
    if length_constraint is not None and getattr(length_constraint, "max_spoken_cost", 0) > 0:
        data["dubbing_window"] = length_constraint.to_prompt_dict()
    return data
