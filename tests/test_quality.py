from __future__ import annotations

import json
from pathlib import Path

from eistara.core.delivery import SubtitleRow
from eistara.core.dubbing import build_audio_mix_plan
from eistara.core.jobs import JsonJobStore, JobStatus, StageName
from eistara.core.jobs.store import STATE_FILE, TASK_FILE
from eistara.core.pipeline import StageContext
from eistara.core.quality import (
    QualityGateService,
    QualitySeverity,
    QualityStageRunner,
    check_audio_mix_plan,
    check_subtitle_rows,
    check_timeline,
    check_translations,
)
from eistara.core.scheduler import SchedulerPolicy, SchedulerService
from eistara.core.timeline import DubTimeline, DubTimelineSegment


def test_check_translations_flags_latin_residue() -> None:
    issues = check_translations({1: "This sentence is still untranslated English text."})

    assert issues[0].code == "translation.latin_residue"
    assert issues[0].severity == QualitySeverity.ERROR


def test_check_subtitle_rows_flags_invalid_time_and_long_text() -> None:
    issues = check_subtitle_rows(
        [
            SubtitleRow(1, 1, "hello", ""),
            SubtitleRow(0, 2, "x" * 50, "y" * 30),
        ]
    )

    assert {issue.code for issue in issues} >= {
        "subtitle.invalid_time",
        "subtitle.source_too_long",
        "subtitle.target_too_long",
    }


def test_check_timeline_flags_missing_audio() -> None:
    timeline = DubTimeline(
        (
            DubTimelineSegment(
                segment_id="1",
                source_start_sec=0,
                source_end_sec=1,
                dub_start_sec=0,
                dub_end_sec=1,
                target_text="hello",
                audio_path=None,
                audio_duration_sec=0,
            ),
        )
    )

    issues = check_timeline(timeline)

    assert {issue.code for issue in issues} >= {"timeline.missing_audio_path", "timeline.empty_audio"}


def test_check_audio_mix_plan_flags_no_clips() -> None:
    timeline = DubTimeline(
        (
            DubTimelineSegment(
                segment_id="1",
                source_start_sec=0,
                source_end_sec=1,
                dub_start_sec=0,
                dub_end_sec=1,
                target_text="hello",
                audio_path=None,
                audio_duration_sec=1,
            ),
        )
    )
    plan = build_audio_mix_plan(timeline, "dub.mp3")

    issues = check_audio_mix_plan(plan)

    assert "audio_mix.no_clips" in {issue.code for issue in issues}


def test_quality_gate_report_passed_false_on_errors() -> None:
    report = QualityGateService().check(translations={1: ""})

    assert report.passed is False
    assert report.error_count == 1


def test_quality_stage_runner_writes_report(tmp_path: Path) -> None:
    translations = tmp_path / "translations.json"
    translations.write_text(json.dumps({"translations": [{"id": 1, "text": ""}]}), encoding="utf-8")
    runner = QualityStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"translations_json": str(translations), "output_dir": str(tmp_path / "output")},
            stage=StageName.QUALITY,
            attempt=1,
        )
    )

    assert result.status == "failed"
    assert Path(result.outputs["quality_report"]).exists()


def test_quality_stage_runner_passes_without_inputs(tmp_path: Path) -> None:
    result = QualityStageRunner().run(StageContext("job", tmp_path, {}, StageName.QUALITY, 1))

    assert result.status == "done"
    assert result.outputs["passed"] is True


def test_scheduler_stops_when_manual_quality_stage_fails(tmp_path: Path) -> None:
    jobs_dir = tmp_path / "jobs"
    job_dir = jobs_dir / "job_0001_quality"
    job_dir.mkdir(parents=True)
    translations = job_dir / "output" / "translations.json"
    translations.parent.mkdir()
    translations.write_text(json.dumps({"translations": [{"id": 1, "text": ""}]}), encoding="utf-8")
    (job_dir / TASK_FILE).write_text(json.dumps({"id": job_dir.name, "translations_json": str(translations)}), encoding="utf-8")
    (job_dir / STATE_FILE).write_text(
        json.dumps(
            {
                "job_id": job_dir.name,
                "status": "pending",
                "current_stage": None,
                "completed_stages": ["download", "transcribe", "translate", "tts_prepare", "tts", "audio_mix", "compose"],
                "failed_stage": None,
                "attempts": {},
                "error": None,
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )

    service = SchedulerService(
        jobs_dir,
        policy=SchedulerPolicy(
            stage_priority=(StageName.QUALITY,),
            stage_worker_limits={StageName.QUALITY: 1},
        ),
    )
    service.job_store.next_stage = lambda state: StageName.QUALITY
    service.register(QualityStageRunner())

    assert service.run_one_ready_stage() is True
    job = JsonJobStore(jobs_dir).load(job_dir.name)
    assert job.state.status == JobStatus.FAILED
    assert job.state.failed_stage == StageName.QUALITY
    assert StageName.QUALITY not in job.state.completed_stages
