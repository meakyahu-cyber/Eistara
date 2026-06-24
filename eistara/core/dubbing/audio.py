from __future__ import annotations

from pathlib import Path

from .models import AudioMixPlan


def process_v1_clip_audio(audio_path: str | Path, plan: AudioMixPlan, gain_db: float = 0.0):
    from pydub import AudioSegment

    audio = AudioSegment.from_file(audio_path).set_channels(1)
    if plan.clip_lowpass_hz > 0:
        audio = audio.low_pass_filter(int(plan.clip_lowpass_hz))
    if (
        plan.clip_peak_normalize_dbfs is not None
        and audio.max_dBFS != float("-inf")
        and audio.max_dBFS > plan.clip_peak_normalize_dbfs
    ):
        audio = audio.apply_gain(float(plan.clip_peak_normalize_dbfs) - audio.max_dBFS)
    if gain_db:
        audio = audio.apply_gain(float(gain_db))

    if plan.clip_tail_cleanup:
        tail_ms = min(max(0, int(plan.clip_tail_cleanup_ms)), len(audio))
        if tail_ms > 0:
            head = audio[:-tail_ms]
            tail = audio[-tail_ms:]
            if plan.clip_tail_cleanup_lowpass_hz > 0:
                tail = tail.low_pass_filter(int(plan.clip_tail_cleanup_lowpass_hz))
            audio = head + tail

    fade_in_ms = min(max(0, int(plan.clip_fade_in_ms)), len(audio))
    fade_out_ms = min(max(0, int(plan.clip_fade_out_ms)), len(audio))
    if fade_in_ms > 0:
        audio = audio.fade_in(fade_in_ms)
    if fade_out_ms > 0:
        audio = audio.fade_out(fade_out_ms)
    tail_pad_ms = max(0, int(plan.clip_tail_pad_ms))
    if tail_pad_ms > 0:
        audio += AudioSegment.silent(duration=tail_pad_ms, frame_rate=audio.frame_rate)
    return audio.set_frame_rate(int(plan.sample_rate_hz))


def v1_processed_clip_duration_sec(audio_path: str | Path, plan: AudioMixPlan) -> float:
    return len(process_v1_clip_audio(audio_path, plan)) / 1000
