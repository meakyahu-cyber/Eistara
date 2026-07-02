from __future__ import annotations

import math
import re


EN_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'-][A-Za-z0-9]+)*")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SPOKEN_COST_TOKEN_RE = re.compile(
    r"\d+(?:[.,]\d+)?%?|[A-Za-z][A-Za-z0-9]*(?:[.'-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[，,、；;：:]|[。！？.!?]"
)
LIGHT_PAUSE_TOKENS = {"，", ",", "、", "；", ";", "：", ":"}
HEAVY_PAUSE_TOKENS = {"。", "！", "？", ".", "!", "?"}


def count_english_words(text: str) -> int:
    return len(EN_WORD_RE.findall(str(text or "")))


def count_chinese_chars(text: str) -> int:
    return len(CJK_RE.findall(str(text or "")))


def estimate_spoken_cost_units(text: str) -> int:
    return int(math.ceil(estimate_spoken_cost(text)))


def estimate_spoken_cost(text: str) -> float:
    cost = 0.0
    for match in SPOKEN_COST_TOKEN_RE.finditer(str(text or "")):
        token = match.group(0)
        if CJK_RE.fullmatch(token):
            cost += 1.0
        elif token in LIGHT_PAUSE_TOKENS:
            cost += 0.7
        elif token in HEAVY_PAUSE_TOKENS:
            cost += 1.0
        elif token[0].isdigit():
            cost += _number_spoken_cost(token)
        else:
            cost += _latin_spoken_cost(token)
    return cost


def _number_spoken_cost(token: str) -> float:
    text = str(token)
    percent = text.endswith("%")
    if percent:
        text = text[:-1]
    digit_count = sum(1 for char in text if char.isdigit())
    decimal_cost = 1 if "." in text else 0
    percent_cost = 3 if percent else 0
    return float(max(1, digit_count + decimal_cost + percent_cost))


def _latin_spoken_cost(token: str) -> float:
    letters = re.findall(r"[A-Za-z]", str(token))
    digits = re.findall(r"\d", str(token))
    if not letters and not digits:
        return 0.0
    if letters and all(letter.isupper() for letter in letters) and len(letters) <= 6:
        letter_cost = len(letters)
    else:
        letter_cost = max(2, math.ceil(len(letters) / 2)) if letters else 0
    return float(max(2, letter_cost + len(digits)))
