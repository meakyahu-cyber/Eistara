from __future__ import annotations

from .models import TranslationItem, TranslationSettings


def split_batches(items: list[TranslationItem], settings: TranslationSettings) -> list[list[TranslationItem]]:
    max_lines = max(1, settings.max_batch_lines)
    max_chars = max(200, settings.max_batch_chars)
    batches: list[list[TranslationItem]] = []
    current: list[TranslationItem] = []
    current_chars = 0

    for item in items:
        item_chars = len(item.source)
        if current and (len(current) >= max_lines or current_chars + item_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars

    if current:
        batches.append(current)
    return batches
