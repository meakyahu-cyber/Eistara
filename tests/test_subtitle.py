from __future__ import annotations

from eistara.core.delivery import ArtifactRole, SubtitleColumn, default_delivery_profile
from eistara.core.subtitle import (
    SubtitleEvent,
    build_display_events,
    format_srt_timestamp,
    parse_srt_timestamp,
    render_srt,
    split_display_text,
    subtitle_visible_len,
)


def test_srt_time_roundtrip() -> None:
    timestamp = format_srt_timestamp(1.234, 65.5)

    assert timestamp == "00:00:01,234 --> 00:01:05,500"
    assert parse_srt_timestamp(timestamp) == (1.234, 65.5)


def test_split_display_text_keeps_words_when_possible() -> None:
    assert split_display_text("hello world from eistara", 12) == ["hello world", "from eistara"]


def test_split_display_text_does_not_cut_latin_word_for_small_overflow() -> None:
    text = "The deadline lands on extraordinarily soon"

    assert split_display_text(text, 32) == ["The deadline lands on extraordinarily", "soon"]


def test_split_display_text_prefers_natural_punctuation_breaks() -> None:
    first = "\u8fd9\u662f\u7b2c\u4e00\u53e5\uff0c"
    second = "\u8fd9\u662f\u7b2c\u4e8c\u53e5\u3002"
    third = "\u8fd9\u662f\u7b2c\u4e09\u53e5"

    assert split_display_text(first + second + third, 10) == [first, second, third]


def test_split_display_text_uses_length_when_no_natural_break_exists() -> None:
    assert split_display_text("abcdefghijklmnop", 6) == ["abcdef", "ghijkl", "mnop"]


def test_split_display_text_splits_long_cjk_token() -> None:
    assert split_display_text("这是一个非常非常长的字幕", 5) == ["这是一个非", "常非常长的", "字幕"]


def test_subtitle_visible_len_ignores_spaces() -> None:
    assert subtitle_visible_len("a b  c") == 3


def test_build_display_events_splits_timeline_by_text_weight() -> None:
    events = build_display_events(
        0,
        4,
        ["这是一个非常非常长的字幕", "short"],
        {"0": 5, "1": 20},
    )

    assert len(events) == 3
    assert events[0].timestamp == "00:00:00,000 --> 00:00:01,667"
    assert events[-1].timestamp == "00:00:03,333 --> 00:00:04,000"


def test_build_display_events_reuses_shorter_column_last_chunk() -> None:
    events = build_display_events(
        0,
        2,
        ["alpha beta gamma", "short"],
        {"0": 5, "1": 20},
    )

    assert [event.lines for event in events] == [
        ("alpha", "short"),
        ("beta", "short"),
        ("gamma", "short"),
    ]


def test_render_srt() -> None:
    text = render_srt([SubtitleEvent(0, 1.5, ("你好", "hello"))])

    assert text == "1\n00:00:00,000 --> 00:00:01,500\n你好\nhello"


def test_delivery_profile_subtitle_columns() -> None:
    profile = default_delivery_profile()

    assert profile.by_role(ArtifactRole.TARGET_SOURCE_SUBTITLE).subtitle_columns == (
        SubtitleColumn.TARGET,
        SubtitleColumn.SOURCE,
    )
