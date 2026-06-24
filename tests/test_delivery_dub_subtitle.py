from __future__ import annotations

from pathlib import Path

from eistara.core.delivery import SubtitleColumn, SubtitleDeliveryGenerator, SubtitleRow


def test_subtitle_delivery_generator_writes_dub_subtitle_from_json(tmp_path: Path) -> None:
    path = tmp_path / "segments.json"
    path.write_text(
        '{"segments":[{"id":"a","start":0,"end":1,"source":"source line","target":"target line","audio_duration_sec":1.25}]}',
        encoding="utf-8",
    )

    written, timeline = SubtitleDeliveryGenerator().write_dub_subtitle_from_json(path, tmp_path)

    assert written == tmp_path / "output_dub.srt"
    assert len(timeline.segments) == 1
    assert written.read_text(encoding="utf-8") == "1\n00:00:00,300 --> 00:00:01,550\ntarget line"
    assert (tmp_path / "output_dub_trans_src.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,300 --> 00:00:01,550\ntarget line\nsource line"
    )


def test_source_timeline_subtitles_keep_original_video_timestamps(tmp_path: Path) -> None:
    rows = [SubtitleRow(start_sec=10, end_sec=12, source="source line", target="target line")]

    written = SubtitleDeliveryGenerator().write_source_timeline_subtitles(rows, tmp_path)

    assert (tmp_path / "src.srt").read_text(encoding="utf-8") == "1\n00:00:10,000 --> 00:00:12,000\nsource line"
    assert (tmp_path / "trans.srt").read_text(encoding="utf-8") == "1\n00:00:10,000 --> 00:00:12,000\ntarget line"
    assert (tmp_path / "trans_src.srt").read_text(encoding="utf-8") == "1\n00:00:10,000 --> 00:00:12,000\ntarget line\nsource line"
    assert (tmp_path / "src_trans.srt").read_text(encoding="utf-8") == "1\n00:00:10,000 --> 00:00:12,000\nsource line\ntarget line"
    assert not (tmp_path / "output_dub.srt").exists()
    assert {path.name for path in written.values()} == {"src.srt", "trans.srt", "trans_src.srt", "src_trans.srt"}


def test_dub_timeline_subtitles_use_dubbed_audio_timestamps(tmp_path: Path) -> None:
    path = tmp_path / "segments.json"
    path.write_text(
        '{"segments":[{"id":"a","start":10,"end":12,"source":"source line","target":"target line","audio_duration_sec":1.25}]}',
        encoding="utf-8",
    )

    written, _timeline = SubtitleDeliveryGenerator().write_dub_subtitle_from_json(path, tmp_path)

    assert written == tmp_path / "output_dub.srt"
    assert (tmp_path / "output_dub.srt").read_text(encoding="utf-8") == "1\n00:00:00,300 --> 00:00:01,550\ntarget line"
    assert (tmp_path / "output_dub_trans_src.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,300 --> 00:00:01,550\ntarget line\nsource line"
    )
    assert not (tmp_path / "src.srt").exists()


def test_subtitle_delivery_generator_splits_long_dub_subtitles_for_display(tmp_path: Path) -> None:
    path = tmp_path / "segments.json"
    path.write_text(
        '{"segments":[{"id":"a","start":0,"end":3,"target":"abcdefghijklmnop","audio_duration_sec":3}]}',
        encoding="utf-8",
    )

    written, _timeline = SubtitleDeliveryGenerator(display_limits={SubtitleColumn.TARGET: 6}).write_dub_subtitle_from_json(
        path,
        tmp_path,
    )

    blocks = written.read_text(encoding="utf-8").split("\n\n")
    assert len(blocks) == 3
    assert all(len(block.splitlines()) == 3 for block in blocks)
    assert [block.splitlines()[2] for block in blocks] == ["abcdef", "ghijkl", "mnop"]
