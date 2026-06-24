from __future__ import annotations

import re


_TERMINAL_BREAKS = set(".!?;" "\u3002\uff01\uff1f\uff1b")
_SOFT_BREAKS = set(",:" "\u3001\uff0c\uff1a")


def subtitle_visible_len(text: str) -> int:
    return sum(1 for char in str(text) if not char.isspace())


def normalize_subtitle_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def split_long_subtitle_token(token: str, max_chars: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for char in str(token):
        candidate = current + char
        if current and subtitle_visible_len(candidate) > max_chars:
            parts.append(current)
            current = char
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts


def split_display_text(text: str, max_chars: int) -> list[str]:
    max_chars = max(1, int(max_chars))
    text = normalize_subtitle_text(text)
    if not text:
        return [""]
    if subtitle_visible_len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    remaining = text
    while subtitle_visible_len(remaining) > max_chars:
        cut = _natural_cut_position(remaining, max_chars)
        chunk = remaining[:cut].strip()
        if not chunk:
            break
        chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks or [text]


def _natural_cut_position(text: str, max_chars: int) -> int:
    terminal: list[int] = []
    soft: list[int] = []
    whitespace: list[int] = []
    visible = 0
    for index, char in enumerate(text):
        if not char.isspace():
            visible += 1
        if visible > max_chars:
            break
        position = index + 1
        if char in _TERMINAL_BREAKS:
            terminal.append(position)
        elif char in _SOFT_BREAKS:
            soft.append(position)
        elif char.isspace():
            whitespace.append(position)

    for candidates in (terminal, soft, whitespace):
        if candidates is whitespace:
            word_boundary = _latin_word_boundary_position(text, max_chars, candidates)
            if word_boundary is not None:
                return word_boundary
        position = _last_readable_candidate(text, candidates, max_chars)
        if position is not None:
            return position
    return _latin_word_boundary_position(text, max_chars, []) or _position_at_visible_len(text, max_chars)


def _latin_word_boundary_position(text: str, max_chars: int, whitespace_candidates: list[int]) -> int | None:
    position = _position_at_visible_len(text, max_chars)
    if position <= 0 or position >= len(text):
        return None
    if not (_is_latin_word_char(text[position - 1]) and _is_latin_word_char(text[position])):
        return None

    word_start = position
    while word_start > 0 and _is_latin_word_char(text[word_start - 1]):
        word_start -= 1
    word_end = position
    while word_end < len(text) and _is_latin_word_char(text[word_end]):
        word_end += 1

    if subtitle_visible_len(text[word_start:word_end]) > max_chars:
        return None

    previous_whitespace = whitespace_candidates[-1] if whitespace_candidates else None
    if previous_whitespace is not None and subtitle_visible_len(text[:previous_whitespace]) >= max(1, int(max_chars * 0.65)):
        return previous_whitespace

    overflow = subtitle_visible_len(text[:word_end]) - max_chars
    if overflow <= max(2, max_chars // 5):
        return word_end
    if previous_whitespace is not None:
        return previous_whitespace
    return None


def _is_latin_word_char(char: str) -> bool:
    return char.isascii() and (char.isalnum() or char in {"'", "-"})


def _last_readable_candidate(text: str, candidates: list[int], max_chars: int) -> int | None:
    if not candidates:
        return None
    min_chars = min(max_chars, max(1, max_chars // 3))
    readable = [position for position in candidates if subtitle_visible_len(text[:position]) >= min_chars]
    return (readable or candidates)[-1]


def _position_at_visible_len(text: str, max_chars: int) -> int:
    visible = 0
    for index, char in enumerate(text):
        if not char.isspace():
            visible += 1
        if visible >= max_chars:
            return index + 1
    return len(text)
