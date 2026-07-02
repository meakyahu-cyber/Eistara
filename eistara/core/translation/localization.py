from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Mapping

from eistara.core.timeline import SourceWindow, build_source_windows

from .models import Terminology, TranslationItem, TranslationSettings
from .pacing import count_chinese_chars, estimate_spoken_cost_units
from .prompting import format_terms


@dataclass(frozen=True, slots=True)
class LocalizationConstraint:
    id: int
    source: str
    draft_text: str
    source_start_sec: float | None
    source_end_sec: float | None
    source_duration_sec: float | None
    window_start_sec: float | None
    window_end_sec: float | None
    window_duration_sec: float | None
    owned_gap_after_sec: float
    seam_gap_sec: float
    usable_window_sec: float | None
    chars_per_sec: float
    max_audio_speed: float
    spoken_cost_per_sec: float
    max_chinese_chars: int
    draft_chinese_chars: int
    min_spoken_cost: int
    max_spoken_cost: int
    draft_spoken_cost: int
    over_limit: bool

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        for key in ("chars_per_sec", "max_chinese_chars", "draft_chinese_chars"):
            data.pop(key, None)
        data["hard_limit_applies"] = self.max_spoken_cost > 0
        return data


@dataclass(frozen=True, slots=True)
class DubbingLengthConstraint:
    id: int
    source: str
    source_start_sec: float | None
    source_end_sec: float | None
    source_duration_sec: float | None
    window_start_sec: float | None
    window_end_sec: float | None
    window_duration_sec: float | None
    owned_gap_after_sec: float
    seam_gap_sec: float
    usable_window_sec: float | None
    chars_per_sec: float
    max_audio_speed: float
    spoken_cost_per_sec: float
    target_chinese_chars: int
    max_chinese_chars: int
    target_spoken_cost: int
    max_spoken_cost: int

    def to_prompt_dict(self) -> dict:
        data = asdict(self)
        for key in ("chars_per_sec", "target_chinese_chars", "max_chinese_chars"):
            data.pop(key, None)
        data["hard_limit_applies"] = self.max_spoken_cost > 0
        return data


def build_dubbing_length_constraints(
    batch: list[TranslationItem],
    settings: TranslationSettings,
    source_windows: Mapping[object, SourceWindow] | None = None,
) -> list[DubbingLengthConstraint]:
    windows = dict(source_windows or build_source_windows(batch, max_gap_after_sec=settings.localization_max_window_gap_sec))
    seam_gap = max(0.0, float(settings.localization_seam_gap_sec))
    chars_per_sec = max(0.001, float(settings.localization_chars_per_sec))
    spoken_cost_per_sec = max(0.001, float(settings.localization_spoken_cost_per_sec))
    max_audio_speed = max(1.0, float(settings.localization_max_audio_speed))
    constraints: list[DubbingLengthConstraint] = []
    for item in batch:
        window = windows.get(item.id)
        window_duration = _positive_float(window.window_duration_sec if window is not None else item.duration_sec)
        usable = max(0.0, window_duration - seam_gap) if window_duration is not None else None
        constraints.append(
            DubbingLengthConstraint(
                id=item.id,
                source=item.source,
                source_start_sec=window.source_start_sec if window is not None else None,
                source_end_sec=window.source_end_sec if window is not None else None,
                source_duration_sec=window.source_duration_sec if window is not None else _positive_float(item.duration_sec),
                window_start_sec=window.window_start_sec if window is not None else None,
                window_end_sec=window.window_end_sec if window is not None else None,
                window_duration_sec=window_duration,
                owned_gap_after_sec=window.owned_gap_after_sec if window is not None else 0.0,
                seam_gap_sec=seam_gap,
                usable_window_sec=usable,
                chars_per_sec=chars_per_sec,
                max_audio_speed=max_audio_speed,
                spoken_cost_per_sec=spoken_cost_per_sec,
                target_chinese_chars=_target_cap(usable, chars_per_sec),
                max_chinese_chars=_hard_cap(usable, chars_per_sec, max_audio_speed),
                target_spoken_cost=_target_cap(usable, spoken_cost_per_sec),
                max_spoken_cost=_hard_cap(usable, spoken_cost_per_sec, max_audio_speed),
            )
        )
    return constraints


def build_localization_constraints(
    batch: list[TranslationItem],
    drafts: Mapping[int, str],
    settings: TranslationSettings,
    source_windows: Mapping[object, SourceWindow] | None = None,
) -> list[LocalizationConstraint]:
    windows = dict(source_windows or build_source_windows(batch, max_gap_after_sec=settings.localization_max_window_gap_sec))
    seam_gap = max(0.0, float(settings.localization_seam_gap_sec))
    chars_per_sec = max(0.001, float(settings.localization_chars_per_sec))
    spoken_cost_per_sec = max(0.001, float(settings.localization_spoken_cost_per_sec))
    max_audio_speed = max(1.0, float(settings.localization_max_audio_speed))
    constraints: list[LocalizationConstraint] = []
    for item in batch:
        window = windows.get(item.id)
        window_duration = _positive_float(window.window_duration_sec if window is not None else item.duration_sec)
        usable = max(0.0, window_duration - seam_gap) if window_duration is not None else None
        draft = str(drafts.get(item.id, ""))
        draft_chars = count_chinese_chars(draft)
        draft_spoken_cost = estimate_spoken_cost_units(draft)
        window_cap = _hard_cap(usable, chars_per_sec, max_audio_speed)
        window_spoken_cap = _hard_cap(usable, spoken_cost_per_sec, max_audio_speed)
        max_chars = _final_cap(window_cap, draft_chars)
        max_spoken_cost = _final_cap(window_spoken_cap, draft_spoken_cost)
        min_spoken_cost = _min_spoken_cost(max_spoken_cost, draft_spoken_cost)
        constraints.append(
            LocalizationConstraint(
                id=item.id,
                source=item.source,
                draft_text=draft,
                source_start_sec=window.source_start_sec if window is not None else None,
                source_end_sec=window.source_end_sec if window is not None else None,
                source_duration_sec=window.source_duration_sec if window is not None else _positive_float(item.duration_sec),
                window_start_sec=window.window_start_sec if window is not None else None,
                window_end_sec=window.window_end_sec if window is not None else None,
                window_duration_sec=window_duration,
                owned_gap_after_sec=window.owned_gap_after_sec if window is not None else 0.0,
                seam_gap_sec=seam_gap,
                usable_window_sec=usable,
                chars_per_sec=chars_per_sec,
                max_audio_speed=max_audio_speed,
                spoken_cost_per_sec=spoken_cost_per_sec,
                max_chinese_chars=max_chars,
                draft_chinese_chars=draft_chars,
                min_spoken_cost=min_spoken_cost,
                max_spoken_cost=max_spoken_cost,
                draft_spoken_cost=draft_spoken_cost,
                over_limit=max_spoken_cost > 0 and draft_spoken_cost > max_spoken_cost,
            )
        )
    return constraints


def build_localization_prompt(
    batch: list[TranslationItem],
    terminology: Terminology,
    settings: TranslationSettings,
    constraints: list[LocalizationConstraint],
) -> str:
    theme = terminology.theme.strip() if settings.use_summary else ""
    terms_text = format_terms(terminology.terms)
    by_id = {item.id: item for item in batch}
    payload = []
    for constraint in constraints:
        item = by_id.get(constraint.id)
        row = constraint.to_prompt_dict()
        if item is not None:
            row.update(
                {
                    "start": item.start,
                    "end": item.end,
                    "speaker": item.speaker,
                }
            )
        payload.append(row)
    batch_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""
You are the second-pass localization editor for a Chinese video dubbing script.

Use the first-pass draft as the base. Rewrite it into concise, natural Chinese spoken narration for TTS.
This pass is a hard timing gate: every hard-limited item must fit its spoken-cost budget.

Hard localization length rule:
1. Each item with hard_limit_applies=true has an independent max_spoken_cost value.
2. For hard-limited items, final_spoken_cost must be between min_spoken_cost and max_spoken_cost, inclusive.
3. Do not output below min_spoken_cost. If the draft is too short, restore omitted source meaning until it reaches the floor.
4. Do not output above max_spoken_cost. If the draft is too long, compress wording while preserving core facts,
   causal links, numbers, names, contrasts, and conclusions.
5. Spoken cost is not just Han characters: count Han characters, Arabic digits, years, percentages, Latin names/acronyms,
   and punctuation pauses because IndexTTS reads or pauses for all of them.
6. The spoken-cost limit never exceeds draft_spoken_cost from the first-pass translation.
7. When timing is available, the limit reserves seam_gap_sec and assumes max_audio_speed as the upper TTS speed.
8. If an item has hard_limit_applies=false, localize naturally without unnecessary expansion.
9. Preserve the exact number of items and exact id values.
10. Return only valid JSON with this schema:
   {{"translations":[{{"id":1,"text":"localized text"}}]}}

Video/theme context:
{theme or "None"}

Terminology:
{terms_text}

Second-pass localization items:
{batch_json}
""".strip()


def build_localization_semantic_review_prompt(
    batch: list[TranslationItem],
    terminology: Terminology,
    settings: TranslationSettings,
    constraints: list[LocalizationConstraint],
    translations: Mapping[int, str],
) -> tuple[str, list[dict]]:
    candidates = localization_semantic_review_items(constraints, translations)
    theme = terminology.theme.strip() if settings.use_summary else ""
    terms_text = format_terms(terminology.terms)
    by_id = {item.id: item for item in batch}
    payload = []
    for row in candidates:
        item = by_id.get(int(row["id"]))
        if item is not None:
            row = {
                **row,
                "start": item.start,
                "end": item.end,
                "speaker": item.speaker,
            }
        payload.append(row)
    review_json = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = f"""
You are a semantic quality gate for a Chinese dubbing localization pass.

Review only whether final_text preserves the meaning of source and draft_text.
Do not complain about harmless compression, word order changes, or natural Chinese phrasing.
Flag only material semantic loss, mistranslation, broken wording, missing causal links, missing enumerated facts,
missing numbers/names, or contradictions.

For each issue, return a concise repair instruction that can be used to rewrite the Chinese line.
If a line is acceptable, omit it from issues.

Return only valid JSON with this schema:
{{"issues":[{{"id":1,"severity":"major","issue_type":"omission","missing_meaning":"...","repair_instruction":"..."}}]}}

Video/theme context:
{theme or "None"}

Terminology:
{terms_text}

Items to review:
{review_json}
""".strip()
    return prompt, candidates


def build_localization_semantic_repair_prompt(
    batch: list[TranslationItem],
    terminology: Terminology,
    settings: TranslationSettings,
    constraints: list[LocalizationConstraint],
    translations: Mapping[int, str],
    issues: list[dict],
) -> str:
    theme = terminology.theme.strip() if settings.use_summary else ""
    terms_text = format_terms(terminology.terms)
    issue_by_id = {int(issue.get("id")): issue for issue in issues if _as_int(issue.get("id")) is not None}
    batch_by_id = {item.id: item for item in batch}
    constraint_by_id = {constraint.id: constraint for constraint in constraints}
    payload = []
    for item_id, issue in issue_by_id.items():
        item = batch_by_id.get(item_id)
        constraint = constraint_by_id.get(item_id)
        if item is None or constraint is None:
            continue
        payload.append(
            {
                "id": item_id,
                "source": constraint.source,
                "draft_text": constraint.draft_text,
                "flawed_final_text": str(translations.get(item_id, "")),
                "repair_instruction": issue.get("repair_instruction") or issue.get("missing_meaning") or "",
                "missing_meaning": issue.get("missing_meaning") or "",
                "min_spoken_cost": constraint.min_spoken_cost,
                "max_spoken_cost": constraint.max_spoken_cost,
                "hard_limit_applies": constraint.max_spoken_cost > 0,
                "start": item.start,
                "end": item.end,
                "speaker": item.speaker,
            }
        )
    repair_json = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""
You are repairing failed lines from a Chinese video dubbing localization pass.

Rewrite each flawed_final_text so it restores the missing source meaning while remaining natural for TTS.
Use draft_text as the base when it is more complete. Preserve numbers, names, causal links, contrasts, and enumerated facts.
Respect the timing budget: for hard-limited items, the repaired text must stay between min_spoken_cost and max_spoken_cost.
Do not add facts that are not in source or draft_text.

Return only valid JSON with this schema:
{{"translations":[{{"id":1,"text":"repaired localized text"}}]}}

Video/theme context:
{theme or "None"}

Terminology:
{terms_text}

Lines to repair:
{repair_json}
""".strip()


def localization_semantic_review_items(
    constraints: list[LocalizationConstraint],
    translations: Mapping[int, str],
    *,
    max_items: int = 12,
) -> list[dict]:
    candidates: list[tuple[int, dict]] = []
    for constraint in constraints:
        final_text = str(translations.get(constraint.id, "")).strip()
        draft_text = constraint.draft_text.strip()
        if not final_text or final_text == draft_text:
            continue
        draft_cost = constraint.draft_spoken_cost
        final_cost = estimate_spoken_cost_units(final_text)
        risk_score = _semantic_risk_score(constraint.source, draft_text, final_text, draft_cost, final_cost)
        if risk_score <= 0:
            continue
        candidates.append(
            (
                risk_score,
                {
                    "id": constraint.id,
                    "source": constraint.source,
                    "draft_text": draft_text,
                    "final_text": final_text,
                    "min_spoken_cost": constraint.min_spoken_cost,
                    "max_spoken_cost": constraint.max_spoken_cost,
                    "draft_spoken_cost": draft_cost,
                    "final_spoken_cost": final_cost,
                },
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in candidates[:max_items]]


def localization_report(
    constraints: list[LocalizationConstraint],
    translations: Mapping[int, str] | None = None,
) -> dict:
    rows = []
    for constraint in constraints:
        text = str((translations or {}).get(constraint.id, ""))
        final_chars = count_chinese_chars(text) if translations is not None else None
        final_spoken_cost = estimate_spoken_cost_units(text) if translations is not None else None
        rows.append(
            {
                **asdict(constraint),
                "hard_limit_applies": constraint.max_spoken_cost > 0,
                "final_chinese_chars": final_chars,
                "final_spoken_cost": final_spoken_cost,
                "final_over_limit": (
                    bool(
                        final_spoken_cost is not None
                        and constraint.max_spoken_cost > 0
                        and final_spoken_cost > constraint.max_spoken_cost
                    )
                ),
                "final_under_min": (
                    bool(
                        final_spoken_cost is not None
                        and constraint.min_spoken_cost > 0
                        and final_spoken_cost < constraint.min_spoken_cost
                    )
                ),
            }
        )
    return {
        "enabled": True,
        "chars_per_sec": constraints[0].chars_per_sec if constraints else None,
        "spoken_cost_per_sec": constraints[0].spoken_cost_per_sec if constraints else None,
        "max_audio_speed": constraints[0].max_audio_speed if constraints else None,
        "seam_gap_sec": constraints[0].seam_gap_sec if constraints else None,
        "items": rows,
        "summary": {
            "item_count": len(rows),
            "timed_item_count": sum(1 for item in rows if item["window_duration_sec"] is not None),
            "hard_limited_item_count": sum(1 for item in rows if item["hard_limit_applies"]),
            "draft_over_limit_count": sum(1 for item in rows if item["over_limit"]),
            "final_over_limit_count": sum(1 for item in rows if item["final_over_limit"]),
            "final_under_min_count": sum(1 for item in rows if item["final_under_min"]),
        },
    }


def _hard_cap(usable_window_sec: float | None, chars_per_sec: float, max_audio_speed: float) -> int:
    if usable_window_sec is None or usable_window_sec <= 0:
        return 0
    capacity = usable_window_sec * chars_per_sec * max_audio_speed
    return max(1, math.floor(capacity))


def _target_cap(usable_window_sec: float | None, chars_per_sec: float) -> int:
    if usable_window_sec is None or usable_window_sec <= 0:
        return 0
    return max(1, math.floor(usable_window_sec * chars_per_sec))


def _final_cap(window_cap: int, draft_chinese_chars: int) -> int:
    draft_cap = max(0, int(draft_chinese_chars))
    if window_cap > 0 and draft_cap > 0:
        return min(window_cap, draft_cap)
    if draft_cap > 0:
        return draft_cap
    return max(0, int(window_cap))


def _min_spoken_cost(max_spoken_cost: int, draft_spoken_cost: int) -> int:
    max_cost = max(0, int(max_spoken_cost))
    draft_cost = max(0, int(draft_spoken_cost))
    required_cut = draft_cost - max_cost
    if max_cost <= 0 or required_cut <= 0:
        return 0
    if max_cost <= 6:
        return max(1, max_cost - 1)
    extra_cut_upper = max(2, math.floor(max_cost * 0.18))
    allowed_extra_cut = min(extra_cut_upper, max(2, math.ceil(required_cut * 0.5)))
    return max(1, max_cost - allowed_extra_cut)


def _semantic_risk_score(
    source: str,
    draft_text: str,
    final_text: str,
    draft_spoken_cost: int,
    final_spoken_cost: int,
) -> int:
    source_lower = source.lower()
    structural_score = 0
    markers = (
        "not only",
        "but also",
        "when we consider",
        "because",
        "therefore",
        "so that",
        "although",
        "while",
        "however",
        "including",
        "such as",
        "for example",
        "in addition",
        "alongside",
    )
    structural_score += sum(2 for marker in markers if marker in source_lower)
    if source.count(",") >= 2 and re.search(r"\b(and|or)\b", source_lower):
        structural_score += 3
    if re.search(r"\b[a-z][a-z-]+,\s+[a-z][a-z-]+,?\s+(and|or)\s+[a-z][a-z-]+\b", source_lower):
        structural_score += 3
    if structural_score <= 0:
        return 0
    score = structural_score
    if draft_spoken_cost - final_spoken_cost >= 6:
        score += 2
    if len(source.split()) >= 24 and len(final_text) < len(draft_text) * 0.82:
        score += 1
    return score


def _as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None
