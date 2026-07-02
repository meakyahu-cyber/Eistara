from __future__ import annotations

import unicodedata


LATIN_SPECIAL_FOLD = {
    "ß": "ss",
    "Ø": "O",
    "ø": "o",
    "Æ": "AE",
    "æ": "ae",
    "Đ": "D",
    "đ": "d",
    "Þ": "Th",
    "þ": "th",
    "Ł": "L",
    "ł": "l",
}


TTS_SYMBOL_DROP = ("\u00ae", "\u2122", "\u00a9")


def fold_latin_diacritics(text: str) -> str:
    out: list[str] = []
    for ch in str(text):
        name = unicodedata.name(ch, "")
        if unicodedata.combining(ch):
            continue
        if name.startswith("LATIN") and "WITH" in name:
            base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
            out.append(base or ch)
        else:
            out.append(ch)
    folded = "".join(out)
    for src, dst in LATIN_SPECIAL_FOLD.items():
        folded = folded.replace(src, dst)
    return folded


def clean_text_for_tts(text: str) -> str:
    cleaned = str(text)
    for char in TTS_SYMBOL_DROP:
        cleaned = cleaned.replace(char, "")
    return fold_latin_diacritics(cleaned).strip()


def is_silent_tts_text(text: str) -> bool:
    return not any(ch.isalnum() for ch in str(text))
