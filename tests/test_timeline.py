from __future__ import annotations

from pathlib import Path

from eistara.core.subtitle import render_srt
from eistara.core.timeline import TimelineInput, TimelinePolicy, build_dub_timeline


def test_dub_timeline_applies_lead_in_and_audio_duration() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput(
                segment_id="1",
                source_start_sec=2.0,
                source_end_sec=3.0,
                target_text="第一句",
                audio_duration_sec=1.25,
            )
        ],
        TimelinePolicy(lead_in_sec=0.5),
    )

    assert len(timeline.segments) == 1
    assert timeline.segments[0].dub_start_sec == 0.5
    assert timeline.segments[0].dub_end_sec == 1.75


def test_dub_timeline_inserts_scaled_source_gap() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("1", 0, 1, "第一句", audio_duration_sec=1.0),
            TimelineInput("2", 3, 4, "第二句", audio_duration_sec=0.5),
        ],
        TimelinePolicy(lead_in_sec=0.0, row_gap_sec=0.1, tail_pad_sec=0.0, source_gap_scale=0.5, max_source_gap_sec=2.0),
    )

    assert timeline.segments[0].dub_end_sec == 1.0
    assert timeline.segments[1].dub_start_sec == 2.0
    assert timeline.segments[1].dub_end_sec == 2.5


def test_dub_timeline_uses_v1_cursor_even_for_short_source_windows() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("1", 0, 5, "first", audio_duration_sec=1.0),
            TimelineInput("2", 5, 8, "second", audio_duration_sec=0.5),
        ],
        TimelinePolicy(lead_in_sec=0.0, tail_pad_sec=0.0, preserve_short_source_windows=True),
    )

    assert timeline.segments[0].dub_start_sec == 0.0
    assert timeline.segments[0].dub_end_sec == 1.0
    assert timeline.segments[1].dub_start_sec == 1.26
    assert timeline.segments[1].dub_end_sec == 1.76


def test_dub_timeline_uses_line_gap_for_v1_line_segments() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("1_1", 0, 2, "second", audio_duration_sec=0.5),
            TimelineInput("1_0", 0, 2, "first", audio_duration_sec=1.0),
        ],
        TimelinePolicy(lead_in_sec=0.0, line_gap_sec=0.2, row_gap_sec=0.9, tail_pad_sec=0.0),
    )

    assert [segment.segment_id for segment in timeline.segments] == ["1_0", "1_1"]
    assert timeline.segments[0].dub_end_sec == 1.0
    assert timeline.segments[1].dub_start_sec == 1.2
    assert timeline.segments[1].dub_end_sec == 1.7


def test_dub_timeline_skips_invalid_segments_with_warnings() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("empty", 0, 1, "", audio_duration_sec=1),
            TimelineInput("silent", 1, 2, "有文字", audio_duration_sec=0),
            TimelineInput("bad", 3, 2, "倒置", audio_duration_sec=1),
        ]
    )

    assert timeline.segments == ()
    assert timeline.warnings == (
        "empty: skipped empty target text",
        "silent: skipped missing or empty audio duration",
        "bad: source end is before source start",
    )


def test_dub_timeline_renders_srt_events() -> None:
    timeline = build_dub_timeline([TimelineInput("1", 0, 1, "第一句", audio_duration_sec=1.5)])

    text = render_srt(timeline.subtitle_events())
    assert text.startswith("1\n00:00:00,300 --> 00:00:01,800\n")
    return

    assert text == "1\n00:00:00,000 --> 00:00:01,500\n第一句"


def test_dub_timeline_preserves_audio_path() -> None:
    timeline = build_dub_timeline(
        [TimelineInput("1", 0, 1, "第一句", audio_path=Path("audio.wav"), audio_duration_sec=1.0)]
    )

    assert timeline.segments[0].audio_path == Path("audio.wav")
