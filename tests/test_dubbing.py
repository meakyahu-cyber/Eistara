from __future__ import annotations

import json
import ast
import wave
from pathlib import Path

import pandas as pd
import pytest

from eistara.core.dubbing import (
    AudioMixPlanStageRunner,
    ComposePlanStageRunner,
    DubbingRenderService,
    build_audio_mix_plan,
)
from eistara.core.media import AudioStreamInfo, MediaInfo
from eistara.core.media import MediaCommandResult
from eistara.core.jobs import StageName
from eistara.core.pipeline import StageContext
from eistara.core.timeline import DubTimeline, DubTimelineSegment, TimelinePolicy, TimelinePreparationService


class FakeRenderer:
    name = "fake"

    def __init__(self) -> None:
        self.audio_calls = []
        self.video_calls = []

    def render_audio_mix(self, plan):
        self.audio_calls.append(plan)
        return MediaCommandResult(("render-audio",), 0)

    def render_video(self, plan):
        self.video_calls.append(plan)
        return MediaCommandResult(("render-video",), 0)


class FakeDurationProbe:
    def __init__(self, duration: float):
        self.duration = duration

    def probe(self, path: str) -> MediaInfo:
        return MediaInfo(path=Path(path), duration_sec=self.duration, audio=AudioStreamInfo(duration_sec=self.duration))


def write_silent_wav(path: Path, duration_ms: int = 1000, sample_rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00" * int(sample_rate * duration_ms / 1000) * 2)
    return path


def test_build_audio_mix_plan_uses_segment_audio_duration() -> None:
    timeline = DubTimeline(
        (
            DubTimelineSegment(
                segment_id="1",
                source_start_sec=0,
                source_end_sec=1,
                dub_start_sec=0.5,
                dub_end_sec=2.5,
                target_text="hello",
                audio_path=Path("a.wav"),
                audio_duration_sec=1.0,
            ),
        )
    )

    plan = build_audio_mix_plan(timeline, "dub.mp3")

    assert len(plan.clips) == 1
    assert plan.clips[0].start_sec == 0.5
    assert plan.clips[0].end_sec == 1.5
    assert plan.duration_sec == 2.5


def test_build_audio_mix_plan_warns_missing_audio_path() -> None:
    timeline = DubTimeline(
        (
            DubTimelineSegment("1", 0, 1, 0, 1, "hello", audio_path=None, audio_duration_sec=1),
        )
    )

    plan = build_audio_mix_plan(timeline, "dub.mp3")

    assert plan.clips == ()
    assert plan.warnings == ("1: skipped missing audio path",)


def test_dubbing_render_plan_includes_compose_plan(tmp_path: Path) -> None:
    timeline = DubTimeline(
        (
            DubTimelineSegment("1", 0, 1, 0, 1, "hello", Path("a.wav"), 1),
        )
    )

    plan = DubbingRenderService().render_plan(timeline, "source.mp4", tmp_path, dub_subtitle="output_dub.srt")

    assert plan.audio_mix.output_audio == tmp_path / "dub.mp3"
    assert plan.compose_video.output_video == tmp_path / "output_dub.mp4"
    assert plan.compose_video.subtitle_path is None
    assert plan.dub_subtitle == Path("output_dub.srt")




def test_audio_mix_plan_stage_runner_writes_plan_and_subtitle(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":1,"target":"hello","audio_path":"a.wav","audio_duration_sec":1}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output")},
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    assert result.outputs["clip_count"] == 1
    assert Path(result.outputs["audio_mix_plan"]).exists()
    assert Path(result.outputs["audio_mix_plan"]) == tmp_path / "output" / "internal" / "audio_mix_plan.json"
    assert Path(result.outputs["dub_subtitles"]).exists()
    assert "dub_audio" not in result.outputs


def test_audio_mix_plan_stage_runner_defaults_to_v1_vocal_only_dub_audio(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":1,"target":"hello","audio_path":"a.wav","audio_duration_sec":1}]}',
        encoding="utf-8",
    )
    background_audio = tmp_path / "output" / "audio" / "background.mp3"
    runner = AudioMixPlanStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output")},
            stage=StageName.AUDIO_MIX,
            attempt=1,
            artifacts={"background_audio": str(background_audio)},
        )
    )

    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["background_audio"] is None


def test_audio_mix_plan_stage_runner_can_opt_into_background_audio_artifact(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":1,"target":"hello","audio_path":"a.wav","audio_duration_sec":1}]}',
        encoding="utf-8",
    )
    background_audio = tmp_path / "output" / "audio" / "background.mp3"
    runner = AudioMixPlanStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "audio_mix_include_background": True,
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
            artifacts={"background_audio": str(background_audio)},
        )
    )

    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["background_audio"] == str(background_audio)


def test_audio_mix_plan_stage_runner_can_render_audio(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.dubbing.runner.is_usable_media_file", lambda *_args, **_kwargs: True)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":1,"target":"hello","audio_path":"a.wav","audio_duration_sec":1}]}',
        encoding="utf-8",
    )
    renderer = FakeRenderer()
    runner = AudioMixPlanStageRunner(renderer=renderer)

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output"), "render_audio": True},
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    assert len(renderer.audio_calls) == 1
    assert result.outputs["audio_render_returncode"] == 0


def test_audio_mix_plan_stage_runner_render_flag_can_be_constructed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.dubbing.runner.is_usable_media_file", lambda *_args, **_kwargs: True)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":1,"target":"hello","audio_path":"a.wav","audio_duration_sec":1}]}',
        encoding="utf-8",
    )
    renderer = FakeRenderer()
    runner = AudioMixPlanStageRunner(renderer=renderer, render_audio=True)

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output")},
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    assert len(renderer.audio_calls) == 1
    assert result.outputs["audio_render_returncode"] == 0


def test_audio_mix_plan_stage_runner_rejects_unreadable_render_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.dubbing.runner.is_usable_media_file", lambda *_args, **_kwargs: False)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":1,"target":"hello","audio_path":"a.wav","audio_duration_sec":1}]}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="audio mix render wrote unreadable audio"):
        AudioMixPlanStageRunner(renderer=FakeRenderer(), render_audio=True).run(
            StageContext(
                job_id="job",
                job_dir=tmp_path,
                task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output")},
                stage=StageName.AUDIO_MIX,
                attempt=1,
            )
        )


def test_audio_mix_plan_stage_runner_skips_without_segments(tmp_path: Path) -> None:
    result = AudioMixPlanStageRunner().run(StageContext("job", tmp_path, {}, StageName.AUDIO_MIX, 1))

    assert result.skipped


def test_audio_mix_plan_stage_runner_prepares_timeline_from_tts_segments(tmp_path: Path) -> None:
    runner = AudioMixPlanStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "output_dir": str(tmp_path / "output"),
                "tts_segments": [
                    {
                        "id": "1",
                        "start": 0,
                        "end": 1,
                        "text": "hello",
                        "output_path": "a.wav",
                        "audio_duration_sec": 1,
                    }
                ],
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    assert result.outputs["clip_count"] == 1
    assert (tmp_path / "output" / "internal" / "dub_segments.json").exists()
    assert Path(result.outputs["dub_segments_json"]) == tmp_path / "output" / "internal" / "dub_segments.json"


def test_audio_mix_plan_stage_runner_writes_v1_new_sub_times(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "lines": ["first", "second"]}]).to_excel(tts_tasks, index=False)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1_0", "start": 0, "end": 2, "target": "first", "audio_path": "1_0.wav", "audio_duration_sec": 1},
                    {"id": "1_1", "start": 0, "end": 2, "target": "second", "audio_path": "1_1.wav", "audio_duration_sec": 0.5},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output"), "tts_tasks": str(tts_tasks)},
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert ast.literal_eval(row["new_sub_times"]) == [[0.3, 1.3], [1.48, 1.98]]
    assert row["timeline_end"] == 2.48
    assert result.outputs["clip_count"] == 2


def test_audio_mix_plan_stage_runner_rebuilds_v1_times_from_processed_audio_duration(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "lines": ["first"]}]).to_excel(tts_tasks, index=False)
    clip = write_silent_wav(tmp_path / "output" / "audio" / "tmp" / "1_0_temp.wav", duration_ms=1000)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "1_0",
                        "start": 0,
                        "end": 2,
                        "target": "first",
                        "audio_path": str(clip),
                        "audio_duration_sec": 3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = AudioMixPlanStageRunner().run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output"), "tts_tasks": str(tts_tasks)},
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert ast.literal_eval(row["new_sub_times"]) == [[0.3, 1.3]]
    assert row["timeline_end"] == 1.8
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["clip_tail_pad_ms"] == 220
    assert plan["clip_tail_pad_counts_in_timeline"] is False
    assert plan["clips"][0]["duration_sec"] == 1.0


def test_audio_mix_plan_stage_runner_can_count_tail_pad_in_timeline(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "lines": ["first"]}]).to_excel(tts_tasks, index=False)
    clip = write_silent_wav(tmp_path / "output" / "audio" / "tmp" / "1_0_temp.wav", duration_ms=1000)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "id": "1_0",
                        "start": 0,
                        "end": 2,
                        "target": "first",
                        "audio_path": str(clip),
                        "audio_duration_sec": 3,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = AudioMixPlanStageRunner(service=DubbingRenderService(clip_tail_pad_counts_in_timeline=True)).run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"dub_segments_json": str(segments_json), "output_dir": str(tmp_path / "output"), "tts_tasks": str(tts_tasks)},
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert ast.literal_eval(row["new_sub_times"]) == [[0.3, 1.52]]
    assert row["timeline_end"] == 2.02
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["clip_tail_pad_counts_in_timeline"] is True
    assert plan["clips"][0]["duration_sec"] == 1.22


def test_audio_mix_plan_stage_runner_applies_v1_global_audio_speed(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "lines": ["long"]}]).to_excel(tts_tasks, index=False)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1_0","start":0,"end":10,"target":"long","audio_path":"1_0.wav","audio_duration_sec":10}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(clip_tail_pad_ms=0, publish_global_audio_speed=True),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(5.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "tts_tasks": str(tts_tasks),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["applied_audio_speed"] == 1.1
    assert report["audio_speed_capped"] is True
    assert report["video_speed_capped"] is True
    assert report["unresolved_retime_overflow_sec"] == pytest.approx(4.335)
    assert report["reason"] == "speeding_dub_and_video_limited_by_caps"
    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert ast.literal_eval(row["new_sub_times"]) == [[0.3, 9.391]]
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["global_audio_speed"] == 1.1
    assert plan["pre_speed_duration_sec"] == 10.8
    assert plan["duration_sec"] == 9.891


def test_audio_mix_plan_source_window_mode_does_not_accelerate_short_dub(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":5,"target":"short","audio_path":"1.wav","audio_duration_sec":5}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(clip_tail_pad_ms=0, publish_global_audio_speed=True),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(10.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["timeline_mode"] == "source_window"
    assert report["applied_audio_speed"] == 1.0
    assert report["final_video_speed"] == 1.0
    assert report["final_dub_duration_sec"] == 10.0
    assert report["reason"] == "source_window_preserves_source_duration"
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["duration_sec"] == 10.0


def test_audio_mix_plan_source_window_stretch_uses_real_tts_duration(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":10,"target":"long","audio_path":"1.wav","audio_duration_sec":10.8}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_target_video_speed_min=0.90,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch_max=1.10,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(10.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["timeline_mode"] == "source_window"
    assert report["max_source_window_overflow_ratio"] == pytest.approx(1.08)
    assert report["requested_source_window_stretch"] == pytest.approx(1.0)
    assert report["applied_source_window_stretch"] == pytest.approx(1.0)
    assert report["source_window_stretch_capped"] is False
    assert report["source_window_stretch_basis_segment_id"] == "1"
    assert report["source_window_required_audio_speed"] == pytest.approx(1.08)
    assert report["applied_audio_speed"] == pytest.approx(1.08)
    assert report["final_dub_duration_sec"] == pytest.approx(10.0)
    assert report["final_video_speed"] == pytest.approx(1.0)

    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["pre_speed_duration_sec"] == pytest.approx(10.8)
    assert plan["duration_sec"] == pytest.approx(10.0)
    assert plan["global_audio_speed"] == pytest.approx(1.08)


def test_audio_mix_plan_source_window_stretch_caps_and_uses_clip_speed(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":10,"target":"too long","audio_path":"1.wav","audio_duration_sec":12}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_target_video_speed_min=0.90,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch_max=1.10,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(10.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["max_source_window_overflow_ratio"] == pytest.approx(1.2)
    assert report["requested_source_window_stretch"] == pytest.approx(1.091)
    assert report["applied_source_window_stretch"] == pytest.approx(1.091)
    assert report["source_window_stretch_capped"] is False
    assert report["source_window_stretch_basis_segment_id"] == "1"
    assert report["source_window_stretch_overflow_sec"] == pytest.approx(0.909)
    assert report["source_window_required_audio_speed"] == pytest.approx(1.1)
    assert report["applied_audio_speed"] == pytest.approx(1.1)
    assert report["source_window_audio_speed_capped"] is False
    assert report["final_video_speed"] == pytest.approx(0.917)
    assert report["unresolved_retime_overflow_sec"] == 0.0

    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["pre_speed_duration_sec"] == pytest.approx(12.0)
    assert plan["duration_sec"] == pytest.approx(10.91)
    assert plan["global_audio_speed"] == pytest.approx(1.1)


def test_audio_mix_plan_retime_keeps_audio_and_video_caps_independent(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":10,"target":"too long","audio_path":"1.wav","audio_duration_sec":15}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_target_video_speed_min=0.90,
            publish_max_audio_speed=1.10,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch_max=1.10,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(10.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["retime_tier"] == 2
    assert report["retime_tier1_max_clip_overflow_after_speed_sec"] == pytest.approx(2.636)
    assert report["requested_source_window_stretch"] == pytest.approx(1.316)
    assert report["applied_source_window_stretch"] == pytest.approx(1.136)
    assert report["source_window_required_audio_speed"] == pytest.approx(1.32)
    assert report["applied_audio_speed"] == pytest.approx(1.14)
    assert report["max_audio_speed"] == pytest.approx(1.14)
    assert report["final_video_speed"] == pytest.approx(0.88)
    assert report["audio_speed_capped"] is True
    assert report["source_window_audio_speed_capped"] is True
    assert report["video_speed_capped"] is False
    assert report["unresolved_retime_overflow_sec"] == pytest.approx(1.798)


def test_audio_mix_plan_retime_tier2_resolves_after_tier1_overflow(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":0,"end":10,"target":"tier2","audio_path":"1.wav","audio_duration_sec":12.7}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_target_video_speed_min=0.90,
            publish_max_audio_speed=1.10,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch_max=1.10,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(10.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["retime_tier"] == 2
    assert report["retime_tier_reason"] == "tier1_local_window_overflow"
    assert report["retime_tier1_max_clip_overflow_after_speed_sec"] > 0
    assert report["applied_source_window_stretch"] == pytest.approx(1.114)
    assert report["applied_audio_speed"] == pytest.approx(1.14)
    assert report["max_clip_overflow_after_speed_sec"] == pytest.approx(0.0)
    assert report["unresolved_retime_overflow_sec"] == pytest.approx(0.0)


def test_audio_mix_plan_last_source_window_uses_trailing_silence(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        '{"segments":[{"id":"1","start":8,"end":10,"target":"closing","audio_path":"1.wav","audio_duration_sec":2.6}]}',
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_target_video_speed_min=0.90,
            publish_max_audio_speed=1.10,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch_max=1.10,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(13.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["requested_source_window_stretch"] == pytest.approx(1.0)
    assert report["applied_source_window_stretch"] == pytest.approx(1.0)
    assert report["applied_audio_speed"] == pytest.approx(1.0)
    assert report["unresolved_retime_overflow_sec"] == 0.0
    assert report["final_video_speed"] == pytest.approx(1.0)

    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["clips"][0]["start_sec"] == pytest.approx(8.0)
    assert plan["clips"][0]["effective_end_sec"] == pytest.approx(10.6)
    assert plan["duration_sec"] == pytest.approx(13.0)


def test_audio_mix_plan_clip_speed_preserves_later_source_window_starts(tmp_path: Path) -> None:
    tts_tasks = tmp_path / "output" / "audio" / "tts_tasks.xlsx"
    tts_tasks.parent.mkdir(parents=True)
    pd.DataFrame([{"number": 1, "lines": ["long"]}, {"number": 2, "lines": ["later"]}]).to_excel(tts_tasks, index=False)
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1_0", "start": 0, "end": 10, "target": "long", "audio_path": "1.wav", "audio_duration_sec": 30},
                    {"id": "2_0", "start": 20, "end": 22, "target": "later", "audio_path": "2.wav", "audio_duration_sec": 1},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_target_video_speed_min=0.90,
            publish_max_audio_speed=1.10,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            source_window_stretch_max=1.10,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(22.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "tts_tasks": str(tts_tasks),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["retime_tier"] == 2
    assert report["applied_source_window_stretch"] == pytest.approx(1.136)
    assert report["applied_audio_speed"] == pytest.approx(1.14)

    rows = pd.read_excel(tts_tasks).to_dict(orient="records")
    assert ast.literal_eval(rows[0]["new_sub_times"]) == [[0.0, 26.316]]
    assert ast.literal_eval(rows[1]["new_sub_times"]) == [[22.72, 23.72]]
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["clips"][0]["speed"] == pytest.approx(1.14)
    assert plan["clips"][1]["speed"] == pytest.approx(1.0)


def test_audio_mix_plan_source_window_borrows_previous_spare_gap_once(tmp_path: Path) -> None:
    segments_json = tmp_path / "segments.json"
    segments_json.write_text(
        json.dumps(
            {
                "segments": [
                    {"id": "1", "start": 0, "end": 3, "target": "first", "audio_path": "1.wav", "audio_duration_sec": 2.0},
                    {"id": "2", "start": 5, "end": 8, "target": "borrow", "audio_path": "2.wav", "audio_duration_sec": 4.0},
                    {"id": "3", "start": 10, "end": 13, "target": "blocked", "audio_path": "3.wav", "audio_duration_sec": 3.5},
                ]
            }
        ),
        encoding="utf-8",
    )
    runner = AudioMixPlanStageRunner(
        service=DubbingRenderService(
            clip_tail_pad_ms=0,
            publish_global_audio_speed=True,
            publish_max_audio_speed=1.10,
        ),
        timeline_policy=TimelinePolicy(
            timeline_mode="source_window",
            lead_in_sec=0.0,
            tail_pad_sec=0.0,
            max_source_gap_sec=0.5,
            source_window_stretch_max=1.0,
            source_window_borrow_max_sec=0.6,
            source_window_borrow_max_ratio=0.5,
            source_window_borrow_min_seam_sec=0.12,
            source_window_retime_tier2_enabled=True,
        ),
        timeline_preparation=TimelinePreparationService(FakeDurationProbe(13.0)),
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "dub_segments_json": str(segments_json),
                "output_dir": str(tmp_path / "output"),
                "raw_audio": str(tmp_path / "raw.wav"),
            },
            stage=StageName.AUDIO_MIX,
            attempt=1,
        )
    )

    report = json.loads(Path(result.outputs["publish_retime_report"]).read_text(encoding="utf-8"))
    assert report["retime_tier"] == 2
    assert report["source_window_borrowed_count"] == 1
    assert report["source_window_borrowed_max_sec"] == pytest.approx(0.419)
    assert report["source_window_borrow_min_seam_sec"] == pytest.approx(0.12)

    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    clips = {clip["segment_id"]: clip for clip in plan["clips"]}
    assert clips["2"]["start_sec"] == pytest.approx(4.696)
    assert clips["2"]["speed"] == pytest.approx(1.0)
    assert clips["2"]["start_sec"] - clips["1"]["effective_end_sec"] >= 0.12
    assert clips["3"]["start_sec"] == pytest.approx(10.23)
    assert clips["3"]["speed"] == pytest.approx(1.14)


def test_compose_plan_stage_runner_writes_plan(tmp_path: Path) -> None:
    runner = ComposePlanStageRunner()

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"source_video": "source.mp4", "output_dir": str(tmp_path / "output")},
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    assert Path(result.outputs["compose_plan"]).exists()
    assert Path(result.outputs["compose_plan"]) == tmp_path / "output" / "internal" / "compose_plan.json"
    plan = json.loads(Path(result.outputs["compose_plan"]).read_text(encoding="utf-8"))
    assert plan["subtitle_path"] is None
    assert plan["external_subtitle_path"].endswith("output_dub.srt")
    assert "dub_video" not in result.outputs


def test_compose_plan_stage_runner_uses_v1_publish_retime_report(tmp_path: Path) -> None:
    report = tmp_path / "output" / "log" / "publish_retime.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"final_video_speed": 1.23}', encoding="utf-8")
    runner = ComposePlanStageRunner(final_loudnorm=True)

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "source_video": "source.mp4",
                "output_dir": str(tmp_path / "output"),
                "publish_retime_report": str(report),
            },
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    plan = json.loads(Path(result.outputs["compose_plan"]).read_text(encoding="utf-8"))
    joined = " ".join(plan["ffmpeg_args"])
    assert plan["video_speed"] == 1.23
    assert "setpts=PTS/1.23000000" in joined
    assert "loudnorm=I=-16:TP=-1.5:LRA=4.5:print_format=none" in joined


def test_compose_plan_stage_runner_normalizes_dub_audio_like_v1(tmp_path: Path) -> None:
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True)
    dub_audio = output_dir / "dub.wav"
    Sine(440).to_audio_segment(duration=500).apply_gain(-12).export(dub_audio, format="wav")

    result = ComposePlanStageRunner().run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "source_video": "source.mp4",
                "output_dir": str(output_dir),
                "dub_audio": str(dub_audio),
            },
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    normalized = output_dir / "normalized_dub.wav"
    assert result.outputs["normalized_dub_audio"] == str(normalized)
    assert normalized.exists()
    plan = json.loads(Path(result.outputs["compose_plan"]).read_text(encoding="utf-8"))
    assert plan["dub_audio"] == str(normalized)


def test_compose_plan_stage_runner_warns_for_v1_video_speed_threshold(tmp_path: Path) -> None:
    report = tmp_path / "output" / "log" / "publish_retime.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"final_video_speed": 1.5}', encoding="utf-8")
    runner = ComposePlanStageRunner(video_speed_warn_min=0.75, video_speed_warn_max=1.35)

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "source_video": "source.mp4",
                "output_dir": str(tmp_path / "output"),
                "publish_retime_report": str(report),
            },
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    assert result.warnings == ["Video retime speed is 1.500x, outside preferred range 0.750-1.350."]


def test_compose_plan_stage_runner_generates_v1_ducked_background_bed(tmp_path: Path) -> None:
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    background = audio_dir / "background.wav"
    Sine(220).to_audio_segment(duration=3000).apply_gain(-9).export(background, format="wav")
    tts_tasks = audio_dir / "tts_tasks.xlsx"
    pd.DataFrame([{"number": 1, "new_sub_times": [[0.5, 1.0]]}]).to_excel(tts_tasks, index=False)
    report = output_dir / "log" / "publish_retime.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"final_video_speed": 1.2}', encoding="utf-8")
    runner = ComposePlanStageRunner(
        background_duck_transition_ms=0,
        background_duck_filter=False,
        background_duck_padding_ms=100,
        background_duck_merge_gap_ms=200,
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "source_video": "source.mp4",
                "output_dir": str(output_dir),
                "tts_tasks": str(tts_tasks),
                "publish_retime_report": str(report),
            },
            stage=StageName.COMPOSE,
            attempt=1,
            artifacts={"background_audio": str(background)},
        )
    )

    assert result.warnings == []
    plan = json.loads(Path(result.outputs["compose_plan"]).read_text(encoding="utf-8"))
    ducked = Path(plan["background_audio"])
    assert ducked.name == "background_ducked.wav"
    assert ducked.exists()
    joined = " ".join(plan["ffmpeg_args"])
    assert "atempo=1.200000" in joined
    assert "amix=inputs=2:duration=longest:dropout_transition=3:normalize=0" in joined


def test_compose_plan_stage_runner_adaptive_background_duck_lifts_weak_separated_bed(tmp_path: Path) -> None:
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    background = audio_dir / "background.wav"
    dub_audio = output_dir / "dub.wav"
    Sine(220).to_audio_segment(duration=3000).apply_gain(-41).export(background, format="wav")
    Sine(440).to_audio_segment(duration=3000).apply_gain(-17).export(dub_audio, format="wav")
    tts_tasks = audio_dir / "tts_tasks.xlsx"
    pd.DataFrame([{"number": 1, "new_sub_times": [[0.5, 2.5]]}]).to_excel(tts_tasks, index=False)
    runner = ComposePlanStageRunner(
        background_duck_transition_ms=0,
        background_duck_filter=True,
        background_duck_lowpass_hz=4200,
        background_duck_target_under_voice_db=16.0,
        background_duck_max_makeup_db=12.0,
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "source_video": "source.mp4",
                "output_dir": str(output_dir),
                "dub_audio": str(dub_audio),
                "tts_tasks": str(tts_tasks),
            },
            stage=StageName.COMPOSE,
            attempt=1,
            artifacts={"background_audio": str(background)},
        )
    )

    assert result.warnings == []
    report = json.loads(Path(result.outputs["background_duck_report"]).read_text(encoding="utf-8"))
    assert report["mode"] == "adaptive"
    assert report["makeup_gain_db"] > 6.0
    assert report["duck_gain_db"] == 0.0
    assert report["filter_enabled"] is False


def test_compose_plan_stage_runner_keeps_source_bed_on_fixed_duck_path(tmp_path: Path) -> None:
    from pydub.generators import Sine

    output_dir = tmp_path / "output"
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True)
    source_bed = audio_dir / "raw.wav"
    Sine(220).to_audio_segment(duration=3000).apply_gain(-9).export(source_bed, format="wav")
    tts_tasks = audio_dir / "tts_tasks.xlsx"
    pd.DataFrame([{"number": 1, "new_sub_times": [[0.5, 1.0]]}]).to_excel(tts_tasks, index=False)
    runner = ComposePlanStageRunner(
        background_bed_mode="source",
        background_duck_adaptive=True,
        background_duck_transition_ms=0,
        source_bed_duck_volume=0.12,
        source_bed_lowpass_hz=3600,
    )

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={
                "source_video": "source.mp4",
                "output_dir": str(output_dir),
                "tts_tasks": str(tts_tasks),
                "raw_audio": str(source_bed),
            },
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    assert result.warnings == ["background_bed_mode uses the original source audio; this can leave source speech under the dub"]
    plan = json.loads(Path(result.outputs["compose_plan"]).read_text(encoding="utf-8"))
    assert Path(plan["background_audio"]).name == "source_bed_ducked.wav"
    assert "background_duck_report" not in result.outputs
    assert not (output_dir / "internal" / "background_duck_report.json").exists()


def test_compose_plan_stage_runner_can_render_video(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.dubbing.runner.is_usable_media_file", lambda *_args, **_kwargs: True)
    renderer = FakeRenderer()
    runner = ComposePlanStageRunner(renderer=renderer)

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"source_video": "source.mp4", "output_dir": str(tmp_path / "output"), "render_video": True},
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    assert len(renderer.video_calls) == 1
    assert result.outputs["video_render_returncode"] == 0


def test_compose_plan_stage_runner_render_flag_can_be_constructed(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.dubbing.runner.is_usable_media_file", lambda *_args, **_kwargs: True)
    renderer = FakeRenderer()
    runner = ComposePlanStageRunner(renderer=renderer, render_video=True)

    result = runner.run(
        StageContext(
            job_id="job",
            job_dir=tmp_path,
            task={"source_video": "source.mp4", "output_dir": str(tmp_path / "output")},
            stage=StageName.COMPOSE,
            attempt=1,
        )
    )

    assert len(renderer.video_calls) == 1
    assert result.outputs["video_render_returncode"] == 0


def test_compose_plan_stage_runner_rejects_unreadable_render_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("eistara.core.dubbing.runner.is_usable_media_file", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="video render wrote unreadable media"):
        ComposePlanStageRunner(renderer=FakeRenderer(), render_video=True).run(
            StageContext(
                job_id="job",
                job_dir=tmp_path,
                task={"source_video": "source.mp4", "output_dir": str(tmp_path / "output")},
                stage=StageName.COMPOSE,
                attempt=1,
            )
        )
