from __future__ import annotations

import re
from typing import Any


TECH_LATIN_ALLOWLIST = {
    "aac",
    "ai",
    "api",
    "asr",
    "av1",
    "chatgpt",
    "cpu",
    "cuda",
    "dns",
    "dram",
    "fps",
    "gb",
    "gbps",
    "gpu",
    "gpt",
    "h264",
    "h265",
    "hd",
    "hdr",
    "http",
    "https",
    "json",
    "kb",
    "llm",
    "mb",
    "mp3",
    "mp4",
    "nvme",
    "pb",
    "pcie",
    "ram",
    "rtx",
    "sata",
    "ssd",
    "srt",
    "tb",
    "tts",
    "usb",
    "vram",
    "wav",
    "youtube",
}

COMMON_ENGLISH_RESIDUE_WORDS = {
    "a",
    "about",
    "after",
    "also",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "have",
    "how",
    "if",
    "in",
    "inside",
    "into",
    "is",
    "it",
    "left",
    "like",
    "long",
    "of",
    "on",
    "or",
    "sentence",
    "still",
    "that",
    "the",
    "this",
    "to",
    "translation",
    "untranslated",
    "was",
    "with",
    "you",
}


def latin_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9]*(?:[-'][A-Za-z0-9]+)*", str(text))


def domain_latin_tokens(text: str) -> set[str]:
    allowed: set[str] = set()
    pattern = r"\b(?:https?://)?[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]*)?"
    for match in re.finditer(pattern, str(text)):
        allowed.update(word.lower() for word in latin_words(match.group(0)))
    return allowed


def technical_latin_tokens(text: str) -> set[str]:
    tokens = {word.lower() for word in latin_words(text) if word.lower() in TECH_LATIN_ALLOWLIST}
    for match in re.finditer(
        r"\b[A-Za-z]{1,8}[- ]?\d{1,5}[A-Za-z0-9-]*\b|\b\d+(?:\.\d+)?\s*[A-Za-z]{1,6}\b",
        str(text),
    ):
        tokens.update(word.lower() for word in latin_words(match.group(0)))
    return tokens


def proper_latin_tokens(text: str) -> set[str]:
    words = latin_words(text)
    allowed: set[str] = set()
    for word in words:
        if word.isupper() and len(word) > 1:
            allowed.add(word.lower())
    for left, right in zip(words, words[1:]):
        if left[:1].isupper() and right[:1].isupper():
            allowed.add(left.lower())
            allowed.add(right.lower())
    return allowed


def source_latin_allowlist(text: str) -> set[str]:
    words = latin_words(text)
    allowed: set[str] = set()
    allowed.update(domain_latin_tokens(text))
    allowed.update(technical_latin_tokens(text))
    for word in words:
        if word.isupper() and len(word) > 1:
            allowed.add(word.lower())
        elif word[:1].isupper():
            allowed.add(word.lower())
        elif any(ch.isdigit() for ch in word):
            allowed.add(word.lower())
    for left, right in zip(words, words[1:]):
        if left[:1].isupper() and right[:1].isupper():
            allowed.add(f"{left}{right}".lower())
    for match in re.finditer(
        r"\b(?:(?:promo|coupon|discount|voucher)\s+code|(?:use|enter|apply|using|with)\s+(?:the\s+)?code)\s+([A-Za-z0-9 ]{1,80})",
        str(text),
        flags=re.IGNORECASE,
    ):
        promo_words = []
        stop_words = {
            "and",
            "for",
            "in",
            "special",
            "discount",
            "at",
            "to",
            "the",
            "below",
            "link",
            "when",
            "on",
            "by",
            "before",
            "after",
            "today",
            "now",
        }
        for token in re.findall(r"[A-Za-z0-9]+", match.group(1)):
            lowered = token.lower()
            if token.isdigit() or (promo_words and lowered in stop_words):
                break
            promo_words.append(token)
        allowed.update(word.lower() for word in promo_words)
    return allowed


def has_excess_latin_text(
    text: str,
    allowed_latin_words: set[str] | None = None,
    target_language: str = "Simplified Chinese",
) -> bool:
    target = target_language.lower()
    if "chinese" not in target and "中文" not in target:
        return False

    allowed = set(TECH_LATIN_ALLOWLIST)
    allowed.update(allowed_latin_words or set())
    allowed.update(domain_latin_tokens(text))
    allowed.update(technical_latin_tokens(text))
    allowed.update(proper_latin_tokens(text))
    words = latin_words(text)
    significant = [word for word in words if word.lower() not in allowed]
    if not significant:
        return False
    if any(len(word) >= 18 for word in significant):
        return True
    latin_chars = sum(len(word) for word in significant)
    if latin_chars >= 32:
        return True
    if len(significant) < 5:
        return False
    lowered = {word.lower() for word in significant}
    return bool(lowered & COMMON_ENGLISH_RESIDUE_WORDS) or any(len(word) >= 7 for word in significant)


def normalize_translation_response(
    response: Any,
    expected_ids: list[int],
    source_latin_by_id: dict[int, set[str]] | None = None,
    target_language: str = "Simplified Chinese",
    enforce_latin: bool = True,
) -> dict[int, str]:
    if not isinstance(response, dict) or "translations" not in response:
        raise ValueError("Missing translations key")

    translations = response["translations"]
    if isinstance(translations, dict):
        iterable = []
        for key, value in translations.items():
            item = dict(value) if isinstance(value, dict) else {"text": value}
            item.setdefault("id", key)
            iterable.append(item)
    elif isinstance(translations, list):
        iterable = translations
    else:
        raise ValueError("translations must be a list or object")

    normalized: dict[int, str] = {}
    for item in iterable:
        if not isinstance(item, dict):
            raise ValueError("Each translation must be an object")
        item_id = int(item.get("id"))
        text = str(item.get("text", "")).replace("\n", " ").strip()
        if not text:
            raise ValueError(f"Empty translation for id {item_id}")
        allowed_latin = (source_latin_by_id or {}).get(item_id, set())
        if enforce_latin and has_excess_latin_text(text, allowed_latin, target_language=target_language):
            raise ValueError(f"Likely untranslated English remains in id {item_id}: {text[:120]}")
        normalized[item_id] = text

    expected = set(expected_ids)
    missing = [item_id for item_id in expected_ids if item_id not in normalized]
    extra = [item_id for item_id in normalized if item_id not in expected]
    if missing or extra:
        raise ValueError(f"Translation id mismatch. missing={missing}, extra={extra}")
    return normalized
