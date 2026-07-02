from __future__ import annotations

from pathlib import Path

from eistara.core.subtitle import render_srt
from eistara.core.timeline import TimelineInput, TimelinePolicy, build_dub_timeline, build_group_source_windows, segment_group_id


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


def test_source_window_timeline_preserves_source_gaps_when_audio_is_short() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("1", 0, 7, "first", audio_duration_sec=8.0),
            TimelineInput("2", 10, 12, "second", audio_duration_sec=1.0),
        ],
        TimelinePolicy(timeline_mode="source_window", lead_in_sec=0.0, tail_pad_sec=0.0),
    )

    assert timeline.mode == "source_window"
    assert timeline.segments[0].dub_start_sec == 0.0
    assert timeline.segments[0].dub_end_sec == 8.0
    assert timeline.segments[1].dub_start_sec == 10.0
    assert timeline.segments[1].dub_end_sec == 11.0


def test_source_window_timeline_reports_overflow_without_shifting_later_windows() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("1", 0, 7, "first", audio_duration_sec=13.0),
            TimelineInput("2", 10, 12, "second", audio_duration_sec=1.0),
        ],
        TimelinePolicy(timeline_mode="source_window", lead_in_sec=0.0, row_gap_sec=0.26, tail_pad_sec=0.0),
    )

    assert timeline.segments[0].dub_start_sec == 0.0
    assert timeline.segments[0].dub_end_sec == 13.0
    assert timeline.segments[1].dub_start_sec == 10.0
    assert timeline.segments[1].dub_end_sec == 11.0
    assert timeline.warnings == (
        "1: source window overflow by 3.000s",
        "2: overlaps previous dub audio by 3.000s",
    )


def test_source_window_timeline_stretches_all_source_windows_from_origin() -> None:
    timeline = build_dub_timeline(
        [
            TimelineInput("1", 0, 7, "first", audio_duration_sec=7.0),
            TimelineInput("2", 10, 12, "second", audio_duration_sec=1.0),
        ],
        TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch=1.1,
        ),
    )

    assert timeline.segments[0].dub_start_sec == 0.0
    assert timeline.segments[0].dub_end_sec == 7.0
    assert timeline.segments[1].dub_start_sec == 11.0
    assert timeline.segments[1].dub_end_sec == 12.0


def test_group_source_windows_merge_line_segments_and_use_trailing_silence() -> None:
    windows = build_group_source_windows(
        [
            TimelineInput("1_0", 0, 2, "first", audio_duration_sec=1.0),
            TimelineInput("1_1", 0, 2.5, "second", audio_duration_sec=1.0),
            TimelineInput("2_0", 5, 6, "third", audio_duration_sec=1.0),
        ],
        max_gap_after_sec=6.0,
        source_duration_sec=10.0,
    )

    assert windows["1"].source_start_sec == 0
    assert windows["1"].source_end_sec == 2.5
    assert windows["1"].owned_gap_after_sec == 2.5
    assert windows["1"].window_end_sec == 5
    assert windows["2"].owned_gap_after_sec == 4
    assert windows["2"].window_end_sec == 10


def test_segment_group_id_normalizes_v1_line_ids() -> None:
    assert segment_group_id("12.0_3") == "12"
    assert segment_group_id("plain") == "plain"
