from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess

from apps.cli.main import cmd_render
from eistara.adapters.media import FfmpegDubbingRenderer, build_audio_mix_ffmpeg_args
from eistara.core.dubbing import AudioClipPlacement, AudioMixPlan


@dataclass(slots=True)
class FakeRunner:
    result: CompletedProcess[str]
    calls: list[tuple[str, ...]]

    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        self.calls.append(args)
        return self.result


def write_silent_wav(path: Path, duration_ms: int = 100, sample_rate: int = 24000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00" * int(sample_rate * duration_ms / 1000) * 2)
    return path


def test_build_audio_mix_ffmpeg_args_delays_and_mixes_clips() -> None:
    plan = AudioMixPlan(
        clips=(
            AudioClipPlacement("1", Path("a.wav"), start_sec=0.5, end_sec=1.5),
            AudioClipPlacement("2", Path("b.wav"), start_sec=1.0, end_sec=2.0, gain_db=-3),
        ),
        output_audio=Path("dub.mp3"),
        duration_sec=2.5,
    )

    args = build_audio_mix_ffmpeg_args(plan, "ffmpeg.exe")
    joined = " ".join(args)

    assert args[0] == "ffmpeg.exe"
    assert "adelay=500|500" in joined
    assert "adelay=1000|1000" in joined
    assert "amix=inputs=2" in joined
    assert args[args.index("-ar") + 1] == "24000"
    assert args[args.index("-b:a") + 1] == "192k"
    assert args[-1] == "dub.mp3"


def test_build_audio_mix_ffmpeg_args_includes_background() -> None:
    plan = AudioMixPlan(
        clips=(AudioClipPlacement("1", Path("a.wav"), 0, 1),),
        output_audio=Path("dub.mp3"),
        duration_sec=1,
        background_audio=Path("bg.wav"),
    )

    args = build_audio_mix_ffmpeg_args(plan)

    assert args.count("-i") == 2
    assert "bg.wav" in args


def test_build_audio_mix_ffmpeg_args_applies_v1_clip_processing() -> None:
    plan = AudioMixPlan(
        clips=(AudioClipPlacement("1", Path("a.wav"), start_sec=0.5, end_sec=1.72),),
        output_audio=Path("dub.mp3"),
        duration_sec=2.5,
        clip_lowpass_hz=6800,
        clip_fade_in_ms=5,
        clip_fade_out_ms=220,
        clip_tail_pad_ms=220,
        clip_tail_pad_counts_in_timeline=True,
        clip_tail_cleanup=True,
        clip_tail_cleanup_ms=420,
        clip_tail_cleanup_lowpass_hz=3600,
    )

    args = build_audio_mix_ffmpeg_args(plan)
    joined = " ".join(args)

    assert "lowpass=f=6800" in joined
    assert "atrim=start=0.580" in joined
    assert "lowpass=f=3600" in joined
    assert "concat=n=2:v=0:a=1" in joined
    assert "afade=t=in:st=0:d=0.005" in joined
    assert "afade=t=out:st=0.780:d=0.220" in joined


def test_build_audio_mix_ffmpeg_args_applies_clip_audio_speed_per_clip() -> None:
    plan = AudioMixPlan(
        clips=(AudioClipPlacement("1", Path("a.wav"), start_sec=0.5, end_sec=10.5, speed=1.10),),
        output_audio=Path("dub.mp3"),
        duration_sec=9.591,
        global_audio_speed=1.0,
    )

    args = build_audio_mix_ffmpeg_args(plan)
    joined = " ".join(args)

    assert "atempo=1.100000[clip0speed]" in joined
    assert "[clip0speed]adelay=500|500" in joined
    assert "[mixpre]atempo" not in joined
    assert "apad=whole_dur=9.791" in joined
    assert args[args.index("-t") + 1] == "9.591"


def test_build_audio_mix_ffmpeg_args_ignores_global_audio_speed_without_clip_speed() -> None:
    plan = AudioMixPlan(
        clips=(AudioClipPlacement("1", Path("a.wav"), start_sec=0.5, end_sec=10.5),),
        output_audio=Path("dub.mp3"),
        duration_sec=9.591,
        global_audio_speed=1.10,
    )

    args = build_audio_mix_ffmpeg_args(plan)
    joined = " ".join(args)

    assert "atempo=1.100000" not in joined


def test_cli_render_audio_mix_preserves_clip_speed_and_tail_pad_mode(tmp_path: Path, capsys) -> None:
    plan = AudioMixPlan(
        clips=(AudioClipPlacement("1", Path("a.wav"), start_sec=0.5, end_sec=1.72, speed=1.10),),
        output_audio=Path("dub.mp3"),
        duration_sec=1.6,
        clip_tail_pad_ms=220,
        clip_tail_pad_counts_in_timeline=True,
    )
    plan_json = tmp_path / "audio_mix_plan.json"
    plan_json.write_text(json.dumps(plan.to_dict(), ensure_ascii=False), encoding="utf-8")

    rc = cmd_render(
        argparse.Namespace(
            render_command="audio-mix",
            plan_json=str(plan_json),
            ffmpeg="ffmpeg",
            dry_run=True,
        )
    )

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    joined = " ".join(output["ffmpeg_args"])
    assert "atempo=1.100000[clip0speed]" in joined
    assert "afade=t=out:st=0.780:d=0.220" in joined


def test_ffmpeg_dubbing_renderer_runs_audio_mix_command(tmp_path: Path) -> None:
    runner = FakeRunner(CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""), [])
    renderer = FfmpegDubbingRenderer(runner=runner, ffmpeg_path="ffmpeg.exe")
    clip = write_silent_wav(tmp_path / "a.wav")
    plan = AudioMixPlan(
        clips=(AudioClipPlacement("1", clip, 0, 1),),
        output_audio=tmp_path / "dub.mp3",
        duration_sec=1,
    )

    result = renderer.render_audio_mix(plan)

    assert result.ok
    assert runner.calls[-1][0] == "ffmpeg.exe"
    assert "-c:a" in runner.calls[-1]
    assert "libmp3lame" in runner.calls[-1]
    assert result.command == runner.calls[-1]


def test_ffmpeg_dubbing_renderer_streams_many_clips_without_long_command(tmp_path: Path) -> None:
    runner = FakeRunner(CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""), [])
    renderer = FfmpegDubbingRenderer(runner=runner, ffmpeg_path="ffmpeg.exe")
    clip = write_silent_wav(tmp_path / "a.wav")
    clips = tuple(
        AudioClipPlacement(str(index), clip, start_sec=index * 0.2, end_sec=index * 0.2 + 0.1)
        for index in range(90)
    )
    plan = AudioMixPlan(clips=clips, output_audio=tmp_path / "dub.mp3", duration_sec=20)

    result = renderer.render_audio_mix(plan)

    assert result.ok
    assert len(runner.calls) == 1
    assert len(" ".join(runner.calls[0])) < 1000
    assert str(clip) not in runner.calls[0]
