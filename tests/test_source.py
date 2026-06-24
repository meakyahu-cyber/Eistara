from __future__ import annotations

from pathlib import Path

from eistara.core.jobs import StageName
from eistara.core.pipeline import StageContext
from eistara.core.source import LocalFileSourceProvider, SourceRequest, SourceSettings, SourceStageRunner


def test_local_file_source_provider_copies_file(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")

    result = LocalFileSourceProvider().acquire(
        SourceRequest(source=str(source), source_type="file", output_dir=tmp_path / "output"),
        SourceSettings(),
    )

    assert result.source_video == tmp_path / "output" / "input.mp4"
    assert result.source_video.read_bytes() == b"video"


def test_source_stage_runner_uses_local_file(tmp_path: Path) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    runner = SourceStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"source": str(source), "source_type": "file", "output_dir": str(tmp_path / "output")},
            stage=StageName.DOWNLOAD,
            attempt=1,
        )
    )

    assert Path(result.outputs["source_video"]).exists()
    assert result.outputs["source_type"] == "file"


def test_source_stage_runner_skips_url_without_provider(tmp_path: Path) -> None:
    result = SourceStageRunner(url_provider=None).run(
        StageContext("job", tmp_path, {"source": "https://example.com/video", "source_type": "url"}, StageName.DOWNLOAD, 1)
    )

    assert result.skipped
    assert result.warnings == ["No source provider for source_type=url"]


def test_source_stage_runner_skips_when_unique_video_already_exists(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "Video Title.webm").write_bytes(b"video")
    (output_dir / "output_dub.mp4").write_bytes(b"rendered")

    result = SourceStageRunner(url_provider=None).run(
        StageContext(
            "job",
            tmp_path,
            {"source": "https://example.com/video", "source_type": "url", "output_dir": str(output_dir)},
            StageName.DOWNLOAD,
            1,
        )
    )

    assert result.skipped
    assert result.outputs["source_video"] == str(output_dir / "Video Title.webm")
    assert result.outputs["source_type"] == "url"
