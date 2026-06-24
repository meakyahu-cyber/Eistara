from __future__ import annotations

import math
import re

from .models import TranslationItem, TranslationPacingBudget, TranslationSettings


EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def count_english_words(text: str) -> int:
    return len(EN_WORD_RE.findall(str(text or "")))


def count_chinese_chars(text: str) -> int:
    return len(CJK_RE.findall(str(text or "")))


def build_pacing_budget(batch: list[TranslationItem], settings: TranslationSettings) -> TranslationPacingBudget:
    source_duration = sum(max(0.0, float(item.duration_sec or 0.0)) for item in batch)
    english_words = sum(count_english_words(item.source) for item in batch)
    estimated_tts_speed = max(0.001, float(settings.estimated_tts_chars_per_sec))
    min_zh_ratio = max(0.001, float(settings.min_zh_chars_per_en_word))
    natural_min_ratio = max(min_zh_ratio, float(settings.natural_min_zh_chars_per_en_word))
    natural_max_ratio = max(natural_min_ratio, float(settings.natural_max_zh_chars_per_en_word))
    soft_pressure_max_ratio = max(min_zh_ratio, float(settings.soft_pressure_max_zh_chars_per_en_word))
    hard_pressure_max_ratio = max(min_zh_ratio, float(settings.hard_pressure_max_zh_chars_per_en_word))
    critical_pressure_max_ratio = max(min_zh_ratio, float(settings.critical_pressure_max_zh_chars_per_en_word))
    estimated_zh_ratio = max(min_zh_ratio, float(settings.estimated_zh_chars_per_en_word))
    prediction_zh_ratio = min(max(estimated_zh_ratio, natural_min_ratio), natural_max_ratio)
    target_pressure = max(0.001, float(settings.target_pacing_pressure))
    soft_pressure = max(target_pressure, float(settings.soft_pacing_pressure))

    if not settings.pacing_budget_enabled:
        return _disabled_budget(
            source_duration,
            english_words,
            prediction_zh_ratio,
            min_zh_ratio,
            natural_min_ratio,
            natural_max_ratio,
            soft_pressure_max_ratio,
            hard_pressure_max_ratio,
            critical_pressure_max_ratio,
            estimated_tts_speed,
            target_pressure,
            soft_pressure,
            "disabled",
        )
    if english_words <= 0:
        return _disabled_budget(
            source_duration,
            english_words,
            prediction_zh_ratio,
            min_zh_ratio,
            natural_min_ratio,
            natural_max_ratio,
            soft_pressure_max_ratio,
            hard_pressure_max_ratio,
            critical_pressure_max_ratio,
            estimated_tts_speed,
            target_pressure,
            soft_pressure,
            "no_english_words",
        )

    english_density = english_words / source_duration if source_duration > 0 else 0.0
    predicted_pressure = english_density * prediction_zh_ratio / estimated_tts_speed
    pressure_at_min_ratio = english_density * min_zh_ratio / estimated_tts_speed
    level = _pressure_level(predicted_pressure, settings) if source_duration > 0 else "unknown"
    if source_duration <= 0:
        return _disabled_budget(
            source_duration,
            english_words,
            prediction_zh_ratio,
            min_zh_ratio,
            natural_min_ratio,
            natural_max_ratio,
            soft_pressure_max_ratio,
            hard_pressure_max_ratio,
            critical_pressure_max_ratio,
            estimated_tts_speed,
            target_pressure,
            soft_pressure,
            "missing_duration_no_pacing_lock",
            level=level,
        )
    if predicted_pressure < target_pressure:
        return _disabled_budget(
            source_duration,
            english_words,
            prediction_zh_ratio,
            min_zh_ratio,
            natural_min_ratio,
            natural_max_ratio,
            soft_pressure_max_ratio,
            hard_pressure_max_ratio,
            critical_pressure_max_ratio,
            estimated_tts_speed,
            target_pressure,
            soft_pressure,
            "natural_language_priority_pressure_below_watch_limit",
            level=level,
        )
    if predicted_pressure < soft_pressure:
        return _disabled_budget(
            source_duration,
            english_words,
            prediction_zh_ratio,
            min_zh_ratio,
            natural_min_ratio,
            natural_max_ratio,
            soft_pressure_max_ratio,
            hard_pressure_max_ratio,
            critical_pressure_max_ratio,
            estimated_tts_speed,
            target_pressure,
            soft_pressure,
            "watch_pressure_avoid_unnecessary_expansion",
            level=level,
        )

    minimum_chars = math.ceil(english_words * min_zh_ratio)
    ideal_min_chars = max(1, math.floor(source_duration * estimated_tts_speed * target_pressure))
    ideal_max_chars = max(ideal_min_chars, math.floor(source_duration * estimated_tts_speed * soft_pressure))
    max_ratio, reason = _ratio_upper_limit(
        predicted_pressure,
        settings,
        soft_pressure_max_ratio,
        hard_pressure_max_ratio,
        critical_pressure_max_ratio,
    )
    max_ratio = max(min_zh_ratio, max_ratio)
    target_ratio = min(natural_max_ratio, max_ratio)
    target_chars = max(minimum_chars, math.ceil(english_words * target_ratio))
    ratio_max_chars = max(target_chars, math.ceil(english_words * max_ratio))
    hard_chars = ratio_max_chars
    if pressure_at_min_ratio > soft_pressure:
        reason = f"{reason}_minimum_ratio_still_high"
    return TranslationPacingBudget(
        enabled=True,
        source_duration_sec=source_duration,
        english_words=english_words,
        english_words_per_sec=english_density,
        estimated_zh_chars_per_en_word=prediction_zh_ratio,
        min_zh_chars_per_en_word=min_zh_ratio,
        natural_min_zh_chars_per_en_word=natural_min_ratio,
        natural_max_zh_chars_per_en_word=natural_max_ratio,
        soft_pressure_max_zh_chars_per_en_word=soft_pressure_max_ratio,
        hard_pressure_max_zh_chars_per_en_word=hard_pressure_max_ratio,
        critical_pressure_max_zh_chars_per_en_word=critical_pressure_max_ratio,
        target_zh_chars_per_en_word=target_ratio,
        max_zh_chars_per_en_word=max_ratio,
        estimated_tts_chars_per_sec=estimated_tts_speed,
        predicted_pressure=predicted_pressure,
        pressure_at_min_zh_ratio=pressure_at_min_ratio,
        target_pressure=target_pressure,
        soft_pressure=soft_pressure,
        ideal_min_zh_chars=ideal_min_chars,
        ideal_max_zh_chars=ideal_max_chars,
        min_zh_chars=minimum_chars,
        target_zh_chars=target_chars,
        hard_zh_chars=hard_chars,
        level=level,
        reason=reason,
    )


def _disabled_budget(
    source_duration: float,
    english_words: int,
    estimated_zh_ratio: float,
    min_zh_ratio: float,
    natural_min_ratio: float,
    natural_max_ratio: float,
    soft_pressure_max_ratio: float,
    hard_pressure_max_ratio: float,
    critical_pressure_max_ratio: float,
    estimated_tts_speed: float,
    target_pressure: float,
    soft_pressure: float,
    reason: str,
    level: str = "disabled",
) -> TranslationPacingBudget:
    english_density = english_words / source_duration if source_duration > 0 else 0.0
    predicted_pressure = english_density * estimated_zh_ratio / estimated_tts_speed if estimated_tts_speed > 0 else 0.0
    pressure_at_min_ratio = english_density * min_zh_ratio / estimated_tts_speed if estimated_tts_speed > 0 else 0.0
    return TranslationPacingBudget(
        enabled=False,
        source_duration_sec=source_duration,
        english_words=english_words,
        english_words_per_sec=english_density,
        estimated_zh_chars_per_en_word=estimated_zh_ratio,
        min_zh_chars_per_en_word=min_zh_ratio,
        natural_min_zh_chars_per_en_word=natural_min_ratio,
        natural_max_zh_chars_per_en_word=natural_max_ratio,
        soft_pressure_max_zh_chars_per_en_word=soft_pressure_max_ratio,
        hard_pressure_max_zh_chars_per_en_word=hard_pressure_max_ratio,
        critical_pressure_max_zh_chars_per_en_word=critical_pressure_max_ratio,
        target_zh_chars_per_en_word=0.0,
        max_zh_chars_per_en_word=0.0,
        estimated_tts_chars_per_sec=estimated_tts_speed,
        predicted_pressure=predicted_pressure,
        pressure_at_min_zh_ratio=pressure_at_min_ratio,
        target_pressure=target_pressure,
        soft_pressure=soft_pressure,
        ideal_min_zh_chars=0,
        ideal_max_zh_chars=0,
        min_zh_chars=0,
        target_zh_chars=0,
        hard_zh_chars=0,
        level=level,
        reason=reason,
    )


def _ratio_upper_limit(
    predicted_pressure: float,
    settings: TranslationSettings,
    soft_pressure_max_ratio: float,
    hard_pressure_max_ratio: float,
    critical_pressure_max_ratio: float,
) -> tuple[float, str]:
    if predicted_pressure >= float(settings.critical_pacing_pressure):
        return critical_pressure_max_ratio, "critical_pressure_ratio_upper_limit"
    if predicted_pressure >= float(settings.hard_pacing_pressure):
        return hard_pressure_max_ratio, "hard_pressure_ratio_upper_limit"
    return soft_pressure_max_ratio, "soft_pressure_ratio_upper_limit"


def _pressure_level(predicted_pressure: float, settings: TranslationSettings) -> str:
    if predicted_pressure >= float(settings.critical_pacing_pressure):
        return "critical"
    if predicted_pressure >= float(settings.hard_pacing_pressure):
        return "hard"
    if predicted_pressure >= float(settings.soft_pacing_pressure):
        return "soft"
    if predicted_pressure >= 1.0:
        return "watch"
    if predicted_pressure < 0.85:
        return "short"
    return "normal"
