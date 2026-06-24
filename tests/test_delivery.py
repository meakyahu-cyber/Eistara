from __future__ import annotations

from pathlib import Path

from eistara.core.delivery import ArtifactRole, DeliveryPublisher, SubtitleColumn, SubtitleDeliveryGenerator, SubtitleRow, default_delivery_profile


def test_default_delivery_profile_roles() -> None:
    profile = default_delivery_profile()

    assert [artifact.role for artifact in profile.artifacts] == [
        ArtifactRole.FINAL_VIDEO,
        ArtifactRole.DUB_SUBTITLE,
        ArtifactRole.DUB_TARGET_SOURCE_SUBTITLE,
        ArtifactRole.TARGET_SUBTITLE,
        ArtifactRole.SOURCE_SUBTITLE,
        ArtifactRole.TARGET_SOURCE_SUBTITLE,
        ArtifactRole.SOURCE_TARGET_SUBTITLE,
    ]
    assert "trans_src.srt" in profile.known_filenames()
    assert "output_dub_trans_src.srt" in profile.known_filenames()


def test_delivery_list_includes_source_aliases(tmp_path: Path) -> None:
    (tmp_path / "trans_src.srt").write_text("target\nsource", encoding="utf-8")
    (tmp_path / "my_video.srt").write_text("target\nsource", encoding="utf-8")

    artifacts = DeliveryPublisher(tmp_path).list_artifacts()
    alias = [item for item in artifacts if item["role"] == ArtifactRole.SOURCE_ALIAS_SUBTITLE.value]

    assert len(alias) == 1
    assert alias[0]["filename"] == "my_video.srt"
    assert alias[0]["source_role"] == ArtifactRole.TARGET_SOURCE_SUBTITLE.value


def test_publish_source_alias_copies_trans_src(tmp_path: Path) -> None:
    (tmp_path / "trans_src.srt").write_text("target\nsource", encoding="utf-8")

    alias = DeliveryPublisher(tmp_path).publish_source_alias(tmp_path / "source_video.webm")

    assert alias == tmp_path / "source_video.srt"
    assert alias.read_text(encoding="utf-8") == "target\nsource"


def test_publish_source_alias_returns_none_without_trans_src(tmp_path: Path) -> None:
    assert DeliveryPublisher(tmp_path).publish_source_alias(tmp_path / "source_video.webm") is None


def test_remove_stale_subtitles(tmp_path: Path) -> None:
    stale = tmp_path / "zh_en.srt"
    stale.write_text("old", encoding="utf-8")

    removed = DeliveryPublisher(tmp_path).remove_stale_subtitles()

    assert removed == [stale]
    assert not stale.exists()


def test_subtitle_delivery_generator_writes_source_timeline_files(tmp_path: Path) -> None:
    rows = [SubtitleRow(start_sec=0, end_sec=1.5, source="hello", target="你好")]

    written = SubtitleDeliveryGenerator().write_source_timeline_subtitles(rows, tmp_path)

    assert set(written) == {
        ArtifactRole.TARGET_SUBTITLE,
        ArtifactRole.SOURCE_SUBTITLE,
        ArtifactRole.TARGET_SOURCE_SUBTITLE,
        ArtifactRole.SOURCE_TARGET_SUBTITLE,
    }
    assert (tmp_path / "trans.srt").read_text(encoding="utf-8").endswith("你好")
    assert (tmp_path / "src.srt").read_text(encoding="utf-8").endswith("hello")
    assert (tmp_path / "trans_src.srt").read_text(encoding="utf-8").endswith("你好\nhello")
    assert (tmp_path / "src_trans.srt").read_text(encoding="utf-8").endswith("hello\n你好")


def test_subtitle_delivery_generator_uses_v1_source_display_limit() -> None:
    generator = SubtitleDeliveryGenerator()

    assert generator.display_limits[SubtitleColumn.SOURCE] == 42
    assert generator.display_limits[SubtitleColumn.TARGET] == 20


def test_subtitle_delivery_generator_reads_display_limits_from_config() -> None:
    generator = SubtitleDeliveryGenerator.from_config(
        {
            "subtitle": {
                "display_source_max_chars_per_line": 40,
                "display_max_chars_per_line": 18,
            }
        }
    )

    assert generator.display_limits[SubtitleColumn.SOURCE] == 40
    assert generator.display_limits[SubtitleColumn.TARGET] == 18


def test_subtitle_delivery_generator_loads_json_rows(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(
        '{"rows":[{"start":0,"end":1,"Source":"hello","Translation":"你好"}]}',
        encoding="utf-8",
    )

    rows = SubtitleDeliveryGenerator().load_rows_json(path)

    assert rows == [SubtitleRow(start_sec=0.0, end_sec=1.0, source="hello", target="你好")]
