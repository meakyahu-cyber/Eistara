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
from eistara.core.timeline import DubTimeline, DubTimelineSegment, TimelinePreparationService


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
    assert ast.literal_eval(row["new_sub_times"]) == [[0.3, 1.52], [1.7, 2.42]]
    assert row["timeline_end"] == 2.92
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
    assert ast.literal_eval(row["new_sub_times"]) == [[0.3, 1.52]]
    assert row["timeline_end"] == 2.02
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
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
    assert report["applied_audio_speed"] == 1.22
    assert report["audio_speed_capped"] is True
    assert report["reason"] == "speeding_dub_to_reach_target_video_speed"
    row = pd.read_excel(tts_tasks).to_dict(orient="records")[0]
    assert ast.literal_eval(row["new_sub_times"]) == [[0.246, 8.443]]
    plan = json.loads(Path(result.outputs["audio_mix_plan"]).read_text(encoding="utf-8"))
    assert plan["global_audio_speed"] == 1.22
    assert plan["pre_speed_duration_sec"] == 10.8
    assert plan["duration_sec"] == 8.852


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
