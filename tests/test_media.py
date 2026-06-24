from __future__ import annotations

from eistara.core.media import MediaInfo, build_audio_extract_plan, build_compose_video_plan


def test_media_info_from_ffprobe_extracts_streams() -> None:
    info = MediaInfo.from_ffprobe(
        "input.mp4",
        {
            "format": {"duration": "12.5", "format_name": "mov,mp4", "bit_rate": "800000"},
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "h264",
                    "width": 1920,
                    "height": 1080,
                    "duration": "12.4",
                    "avg_frame_rate": "30000/1001",
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac",
                    "channels": 2,
                    "sample_rate": "44100",
                    "duration": "12.5",
                },
            ],
        },
    )

    assert info.has_video
    assert info.has_audio
    assert info.duration_sec == 12.5
    assert info.bit_rate == 800000
    assert info.video is not None
    assert info.video.codec == "h264"
    assert info.video.width == 1920
    assert round(info.video.frame_rate or 0, 3) == 29.97
    assert info.audio is not None
    assert info.audio.sample_rate_hz == 44100


def test_audio_extract_plan_builds_ffmpeg_args() -> None:
    plan = build_audio_extract_plan("source.mp4", "raw.wav")

    assert plan.ffmpeg_args("ffmpeg.exe") == (
        "ffmpeg.exe",
        "-y",
        "-i",
        "source.mp4",
        "-vn",
        "-ac",
        "2",
        "-ar",
        "44100",
        "raw.wav",
    )


def test_audio_extract_plan_can_build_v1_asr_mp3_args() -> None:
    plan = build_audio_extract_plan(
        "source.mp4",
        "raw.mp3",
        sample_rate_hz=16000,
        channels=1,
        audio_codec="libmp3lame",
        audio_bitrate="32k",
        metadata={"encoding": "UTF-8"},
    )

    assert plan.ffmpeg_args("ffmpeg.exe") == (
        "ffmpeg.exe",
        "-y",
        "-i",
        "source.mp4",
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "32k",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-metadata",
        "encoding=UTF-8",
        "raw.mp3",
    )


def test_compose_video_plan_copies_video_without_subtitles() -> None:
    plan = build_compose_video_plan("source.mp4", "dub.wav", "out.mp4")

    args = plan.ffmpeg_args()

    assert "-c:v" in args
    assert args[args.index("-c:v") + 1] == "copy"
    assert args[args.index("-b:a") + 1] == "192k"
    assert args[args.index("-ar") + 1] == "48000"
    assert args[args.index("-ac") + 1] == "2"
    assert args[-1] == "out.mp4"


def test_compose_video_plan_reencodes_when_burning_subtitles() -> None:
    plan = build_compose_video_plan("source.mp4", "dub.wav", "out.mp4", "output_dub.srt")

    args = plan.ffmpeg_args()

    assert "-vf" in args
    assert args[args.index("-c:v") + 1] == "libx264"


def test_compose_video_plan_uses_configured_gpu_encoder_and_audio_settings() -> None:
    plan = build_compose_video_plan(
        "source.mp4",
        "dub.wav",
        "out.mp4",
        video_encoder="h264_nvenc",
        audio_bitrate="256k",
        audio_sample_rate_hz=44100,
        audio_channels=1,
    )

    args = plan.ffmpeg_args()

    assert args[args.index("-c:v") + 1] == "h264_nvenc"
    assert args[args.index("-b:a") + 1] == "256k"
    assert args[args.index("-ar") + 1] == "44100"
    assert args[args.index("-ac") + 1] == "1"


def test_compose_video_plan_retimes_video_and_applies_final_audio_filters() -> None:
    plan = build_compose_video_plan(
        "source.mp4",
        "dub.wav",
        "out.mp4",
        video_encoder="h264_nvenc",
        video_speed=1.25,
        source_fps=29.97,
        final_loudnorm=True,
        final_loudnorm_i=-16,
        final_loudnorm_tp=-1.5,
        final_loudnorm_lra=4.5,
        final_smooth=True,
    )

    args = plan.ffmpeg_args()
    joined = " ".join(args)

    assert "-filter_complex" in args
    assert "setpts=PTS/1.25000000" in joined
    assert "fps=29.970000" in joined
    assert "acompressor=" in joined
    assert "loudnorm=I=-16:TP=-1.5:LRA=4.5:print_format=none" in joined
    assert "[v]" in args
    assert "[a]" in args
    assert args[args.index("-c:v") + 1] == "h264_nvenc"


def test_compose_video_plan_mixes_v1_background_bed_with_dub_audio() -> None:
    plan = build_compose_video_plan(
        "source.mp4",
        "dub.wav",
        "out.mp4",
        background_audio="background_ducked.wav",
        video_speed=1.25,
        final_loudnorm=True,
    )

    args = plan.ffmpeg_args()
    joined = " ".join(args)

    assert args.count("-i") == 3
    assert "background_ducked.wav" in args
    assert "[1:a]atempo=1.250000[bg]" in joined
    assert "[bg][2:a]amix=inputs=2:duration=longest:dropout_transition=3:normalize=0[mix]" in joined
    assert "loudnorm=I=-16:TP=-1.5:LRA=4.5:print_format=none" in joined
    assert "[v]" in args
    assert "[a]" in args
