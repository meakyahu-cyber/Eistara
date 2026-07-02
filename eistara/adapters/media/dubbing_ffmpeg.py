from __future__ import annotations

import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path

from eistara.core.dubbing import AudioMixPlan
from eistara.core.dubbing.audio import process_v1_clip_audio
from eistara.core.dubbing.renderers import DubbingRenderer
from eistara.core.media import ComposeVideoPlan, MediaCommandResult

from .ffmpeg import FfmpegMediaProvider, ProcessRunner


@dataclass(slots=True)
class FfmpegDubbingRenderer(DubbingRenderer):
    runner: ProcessRunner | None = None
    ffmpeg_path: str = "ffmpeg"
    name: str = "ffmpeg-dubbing"

    def __post_init__(self) -> None:
        self.media = FfmpegMediaProvider(runner=self.runner, ffmpeg_path=self.ffmpeg_path)

    def render_audio_mix(self, plan: AudioMixPlan) -> MediaCommandResult:
        plan.output_audio.parent.mkdir(parents=True, exist_ok=True)
        if plan.background_audio is not None:
            return self.media._run_command(build_audio_mix_ffmpeg_args(plan, self.ffmpeg_path))
        return _render_v1_streamed_audio_mix(plan, self.media, self.ffmpeg_path)

    def render_video(self, plan: ComposeVideoPlan) -> MediaCommandResult:
        return self.media.compose_video(plan)


def build_audio_mix_ffmpeg_args(plan: AudioMixPlan, executable: str = "ffmpeg") -> tuple[str, ...]:
    args = [executable, "-y"]
    input_labels: list[str] = []
    filters: list[str] = []
    mix_duration_sec = plan.pre_speed_duration_sec or plan.duration_sec

    if plan.background_audio is not None:
        args.extend(["-i", str(plan.background_audio)])
        filters.append(f"[0:a]volume={_db_expr(plan.background_gain_db)}[bg]")
        input_labels.append("[bg]")

    for index, clip in enumerate(plan.clips):
        input_index = index + (1 if plan.background_audio is not None else 0)
        args.extend(["-i", str(clip.audio_path)])
        delay_ms = max(0, int(round(clip.start_sec * 1000)))
        label = f"clip{index}"
        input_label = f"[{input_index}:a]"
        processed_label = _append_v1_clip_filters(filters, input_label, f"clip{index}", clip, plan)
        clip_speed = _clip_speed(clip)
        if clip_speed > 1.001:
            speed_label = f"{label}speed"
            filters.append(f"{processed_label}{_atempo_filter(clip_speed)}[{speed_label}]")
            processed_label = f"[{speed_label}]"
        effective_clip_end = float(clip.start_sec) + (_clip_render_duration_sec(clip, plan) / clip_speed)
        filters.append(
            f"{processed_label}adelay={delay_ms}|{delay_ms},"
            f"apad=whole_dur={max(mix_duration_sec, effective_clip_end):.3f}"
            f"[{label}]"
        )
        input_labels.append(f"[{label}]")

    if not input_labels:
        raise ValueError("Audio mix plan has no inputs")

    if len(input_labels) == 1:
        filters.append(f"{input_labels[0]}anull[mixpre]")
    else:
        filters.append(f"{''.join(input_labels)}amix=inputs={len(input_labels)}:duration=longest:dropout_transition=0[mixpre]")

    filters.append("[mixpre]anull[mix]")

    args.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[mix]",
            "-t",
            f"{max(0.001, plan.duration_sec):.3f}",
            "-ac",
            str(plan.channels),
            "-ar",
            str(plan.sample_rate_hz),
            "-b:a",
            str(plan.bitrate),
            str(plan.output_audio),
        ]
    )
    return tuple(args)


def _render_v1_streamed_audio_mix(plan: AudioMixPlan, media: FfmpegMediaProvider, executable: str) -> MediaCommandResult:
    temp_paths: list[Path] = []
    merge_path = _new_temp_wav(plan.output_audio.parent)
    temp_paths.append(merge_path)
    try:
        _write_v1_streamed_mix_wav(plan, merge_path, executable)
        return media._run_command(_v1_export_audio_args(executable, merge_path, plan.output_audio, plan.bitrate))
    finally:
        for path in temp_paths:
            try:
                path.unlink()
            except OSError:
                pass


def _new_temp_wav(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=directory)
    try:
        return Path(handle.name)
    finally:
        handle.close()


def _write_v1_streamed_mix_wav(plan: AudioMixPlan, output_file: Path, executable: str = "ffmpeg") -> None:
    if _plan_has_clip_overlap(plan):
        _write_v1_overlay_mix_wav(plan, output_file, executable)
        return

    sample_rate = int(plan.sample_rate_hz)
    channels = 1
    sample_width = 2
    with wave.open(str(output_file), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        previous_end: float | None = None
        for clip in plan.clips:
            if not clip.audio_path.exists():
                continue
            start_time = float(clip.start_sec)
            if previous_end is None:
                silence_duration = start_time
            else:
                silence_duration = start_time - previous_end
            if silence_duration > 0:
                _write_silence(wav_file, int(silence_duration * 1000), channels, sample_width)
            audio_segment = _process_v1_clip_audio(clip.audio_path, plan, clip.gain_db)
            audio_segment = _speed_audio_segment(audio_segment, _clip_speed(clip), executable)
            _write_audio_segment(wav_file, audio_segment, sample_rate, channels, sample_width)
            previous_end = start_time + (len(audio_segment) / 1000)
        if previous_end is not None and float(plan.duration_sec) > previous_end:
            _write_silence(wav_file, int((float(plan.duration_sec) - previous_end) * 1000), channels, sample_width)


def _plan_has_clip_overlap(plan: AudioMixPlan) -> bool:
    previous_end: float | None = None
    for clip in plan.clips:
        current_start = float(clip.start_sec)
        if previous_end is not None and current_start < previous_end - 0.001:
            return True
        previous_end = max(previous_end or 0.0, float(clip.start_sec) + _clip_render_duration_sec(clip, plan) / _clip_speed(clip))
    return False


def _write_v1_overlay_mix_wav(plan: AudioMixPlan, output_file: Path, executable: str = "ffmpeg") -> None:
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise RuntimeError(f"overlap audio mix requires pydub: {exc}") from exc

    sample_rate = int(plan.sample_rate_hz)
    channels = 1
    sample_width = 2
    duration_ms = int(
        round(
            max(
                float(plan.duration_sec),
                max((float(clip.start_sec) + _clip_render_duration_sec(clip, plan) / _clip_speed(clip) for clip in plan.clips), default=0.0),
            )
            * 1000
        )
    )
    mix = AudioSegment.silent(duration=max(1, duration_ms), frame_rate=sample_rate).set_channels(channels).set_sample_width(sample_width)
    for clip in plan.clips:
        if not clip.audio_path.exists():
            continue
        audio_segment = _process_v1_clip_audio(clip.audio_path, plan, clip.gain_db)
        audio_segment = _speed_audio_segment(audio_segment, _clip_speed(clip), executable)
        audio_segment = audio_segment.set_frame_rate(sample_rate).set_channels(channels).set_sample_width(sample_width)
        mix = mix.overlay(audio_segment, position=max(0, int(round(float(clip.start_sec) * 1000))))
    mix.export(output_file, format="wav")


def _clip_speed(clip) -> float:
    return max(1.0, float(getattr(clip, "speed", 1.0) or 1.0))


def _clip_render_duration_sec(clip, plan: AudioMixPlan) -> float:
    duration = max(0.001, float(clip.duration_sec))
    if bool(getattr(plan, "clip_tail_pad_counts_in_timeline", False)):
        return duration
    return duration + max(0.0, float(plan.clip_tail_pad_ms) / 1000)


def _process_v1_clip_audio(audio_path: Path, plan: AudioMixPlan, gain_db: float):
    return process_v1_clip_audio(audio_path, plan, gain_db=gain_db)


def _speed_audio_segment(audio_segment, speed: float, executable: str = "ffmpeg"):
    speed = float(speed)
    if speed <= 1.001:
        return audio_segment

    import subprocess

    input_path = _new_temp_wav(Path(tempfile.gettempdir()))
    output_path = _new_temp_wav(Path(tempfile.gettempdir()))
    try:
        audio_segment.export(input_path, format="wav")
        command = (
            executable,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-filter:a",
            _atempo_filter(speed),
            str(output_path),
        )
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(f"ffmpeg atempo failed for clip speed {speed:.6f}: {detail}")

        from pydub import AudioSegment

        return AudioSegment.from_file(output_path)
    finally:
        for path in (input_path, output_path):
            try:
                path.unlink()
            except OSError:
                pass


def _write_silence(wav_file, duration_ms: int, channels: int, sample_width: int, chunk_ms: int = 30000) -> None:
    remaining_ms = max(0, int(duration_ms))
    sample_rate = wav_file.getframerate()
    while remaining_ms > 0:
        current_ms = min(remaining_ms, chunk_ms)
        frame_count = round(sample_rate * current_ms / 1000)
        wav_file.writeframes(b"\x00" * frame_count * channels * sample_width)
        remaining_ms -= current_ms


def _write_audio_segment(wav_file, audio_segment, sample_rate: int, channels: int, sample_width: int) -> None:
    audio_segment = audio_segment.set_frame_rate(sample_rate).set_channels(channels).set_sample_width(sample_width)
    wav_file.writeframes(audio_segment.raw_data)


def _v1_export_audio_args(executable: str, input_file: Path, output_file: Path, bitrate: str) -> tuple[str, ...]:
    args = [
        executable,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_file),
    ]
    if output_file.suffix.lower() == ".mp3":
        args.extend(["-c:a", "libmp3lame", "-b:a", str(bitrate)])
    args.append(str(output_file))
    return tuple(args)


def _db_expr(db: float) -> str:
    if db == 0:
        return "1.0"
    return f"{float(db):.3f}dB"


def _atempo_filter(speed: float) -> str:
    factors: list[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return ",".join(f"atempo={factor:.6f}" for factor in factors)


def _append_v1_clip_filters(filters: list[str], input_label: str, prefix: str, clip, plan: AudioMixPlan) -> str:
    raw_duration_sec = max(0.001, float(clip.duration_sec))
    if bool(getattr(plan, "clip_tail_pad_counts_in_timeline", False)):
        raw_duration_sec = max(0.001, raw_duration_sec - max(0.0, plan.clip_tail_pad_ms / 1000))
    chain = [f"volume={_db_expr(clip.gain_db)}"]
    if plan.clip_lowpass_hz > 0:
        chain.append(f"lowpass=f={int(plan.clip_lowpass_hz)}")

    current_label = input_label
    tail_cleanup_sec = max(0.0, plan.clip_tail_cleanup_ms / 1000)
    if plan.clip_tail_cleanup and plan.clip_tail_cleanup_lowpass_hz > 0 and tail_cleanup_sec > 0:
        tail_start = max(0.0, raw_duration_sec - tail_cleanup_sec)
        if tail_start > 0.001:
            head_in = f"{prefix}headin"
            tail_in = f"{prefix}tailin"
            head = f"{prefix}head"
            tail = f"{prefix}tail"
            current = f"{prefix}tailclean"
            filters.append(f"{current_label}{','.join(chain)},asplit=2[{head_in}][{tail_in}]")
            filters.append(f"[{head_in}]atrim=end={tail_start:.3f},asetpts=PTS-STARTPTS[{head}]")
            filters.append(
                f"[{tail_in}]atrim=start={tail_start:.3f},asetpts=PTS-STARTPTS,"
                f"lowpass=f={int(plan.clip_tail_cleanup_lowpass_hz)}[{tail}]"
            )
            filters.append(f"[{head}][{tail}]concat=n=2:v=0:a=1[{current}]")
            current_label = f"[{current}]"
            chain = []
        else:
            chain.append(f"lowpass=f={int(plan.clip_tail_cleanup_lowpass_hz)}")

    fade_in_sec = max(0.0, plan.clip_fade_in_ms / 1000)
    if fade_in_sec > 0:
        chain.append(f"afade=t=in:st=0:d={min(fade_in_sec, raw_duration_sec):.3f}")

    fade_out_sec = max(0.0, min(plan.clip_fade_out_ms / 1000, raw_duration_sec))
    if fade_out_sec > 0:
        fade_start = max(0.0, raw_duration_sec - fade_out_sec)
        chain.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out_sec:.3f}")

    if not chain:
        return current_label
    output = f"{prefix}processed"
    filters.append(f"{current_label}{','.join(chain)}[{output}]")
    return f"[{output}]"
