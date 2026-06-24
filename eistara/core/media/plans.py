from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class MediaCommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class AudioExtractPlan:
    source_video: Path
    output_audio: Path
    sample_rate_hz: int = 44100
    channels: int = 2
    audio_codec: str = ""
    audio_bitrate: str = ""
    output_format: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)
    overwrite: bool = True

    def ffmpeg_args(self, executable: str = "ffmpeg") -> tuple[str, ...]:
        args = [executable]
        if self.overwrite:
            args.append("-y")
        args.extend(
            [
                "-i",
                str(self.source_video),
                "-vn",
            ]
        )
        if self.audio_codec:
            args.extend(["-c:a", self.audio_codec])
        if self.audio_bitrate:
            args.extend(["-b:a", self.audio_bitrate])
        if self.audio_codec:
            args.extend(["-ar", str(self.sample_rate_hz), "-ac", str(self.channels)])
        else:
            args.extend(["-ac", str(self.channels), "-ar", str(self.sample_rate_hz)])
        for key, value in self.metadata.items():
            args.extend(["-metadata", f"{key}={value}"])
        if self.output_format:
            args.extend(["-f", self.output_format])
        args.append(str(self.output_audio))
        return tuple(args)


@dataclass(frozen=True, slots=True)
class ComposeVideoPlan:
    source_video: Path
    dub_audio: Path
    output_video: Path
    subtitle_path: Path | None = None
    background_audio: Path | None = None
    copy_video: bool = True
    video_encoder: str = ""
    audio_bitrate: str = "192k"
    audio_sample_rate_hz: int = 48000
    audio_channels: int = 2
    video_speed: float = 1.0
    video_retime_fps_mode: str = "cfr"
    source_fps: float = 30.0
    final_loudnorm: bool = False
    final_loudnorm_i: float = -16.0
    final_loudnorm_tp: float = -1.5
    final_loudnorm_lra: float = 4.5
    final_smooth: bool = False
    final_smooth_threshold_db: float = -30.0
    final_smooth_ratio: float = 4.0
    final_smooth_attack_ms: float = 6.0
    final_smooth_release_ms: float = 550.0
    final_smooth_makeup_db: float = 5.5
    background_audio_speed: float = 1.0
    background_mix_duration: str = "longest"
    overwrite: bool = True

    def ffmpeg_args(self, executable: str = "ffmpeg") -> tuple[str, ...]:
        args = [executable]
        if self.overwrite:
            args.append("-y")
        args.extend(["-i", str(self.source_video)])
        if self.background_audio is not None:
            args.extend(["-i", str(self.background_audio)])
        args.extend(["-i", str(self.dub_audio)])
        filter_complex, video_map, audio_map = self._filter_complex()
        if filter_complex:
            args.extend(["-filter_complex", filter_complex, "-map", video_map, "-map", audio_map])
        else:
            if self.subtitle_path is not None:
                args.extend(["-vf", f"subtitles={self.subtitle_path.as_posix()}"])
            args.extend(["-map", "0:v:0", "-map", "1:a:0"])
        needs_video_encode = self.subtitle_path is not None or abs(float(self.video_speed) - 1.0) > 0.001
        video_codec = self.video_encoder or ("copy" if self.copy_video and not needs_video_encode else "libx264")
        args.extend(["-c:v", video_codec])
        args.extend(
            [
                "-c:a",
                "aac",
                "-b:a",
                str(self.audio_bitrate),
                "-ar",
                str(self.audio_sample_rate_hz),
                "-ac",
                str(self.audio_channels),
                "-shortest",
                str(self.output_video),
            ]
        )
        return tuple(args)

    def _filter_complex(self) -> tuple[str, str, str] | tuple[None, str, str]:
        video_filters = self._video_filters()
        audio_filters = self._audio_filters(force_format=self.background_audio is not None)
        if self.subtitle_path is not None and video_filters == [f"subtitles={self.subtitle_path.as_posix()}"] and not audio_filters:
            return None, "0:v:0", "1:a:0"
        if self.background_audio is None and not video_filters and not audio_filters:
            return None, "0:v:0", "1:a:0"

        filters: list[str] = []
        video_map = "0:v:0"
        audio_map = "2:a:0" if self.background_audio is not None else "1:a:0"
        if video_filters:
            filters.append(f"[0:v]{','.join(video_filters)}[v]")
            video_map = "[v]"
        if self.background_audio is not None:
            bg_filters = _atempo_filters(self.background_audio_speed) if abs(float(self.background_audio_speed) - 1.0) > 0.001 else ["anull"]
            filters.append(f"[1:a]{','.join(bg_filters)}[bg]")
            mix_duration = self.background_mix_duration if self.background_mix_duration in {"first", "longest", "shortest"} else "longest"
            filters.append(
                f"[bg][2:a]amix=inputs=2:duration={mix_duration}:dropout_transition=3:normalize=0[mix]"
            )
            if audio_filters:
                filters.append(f"[mix]{','.join(audio_filters)}[a]")
            else:
                filters.append("[mix]anull[a]")
            audio_map = "[a]"
        elif audio_filters:
            filters.append(f"[1:a]{','.join(audio_filters)}[a]")
            audio_map = "[a]"
        return ";".join(filters), video_map, audio_map

    def _video_filters(self) -> list[str]:
        filters: list[str] = []
        speed = float(self.video_speed)
        if abs(speed - 1.0) > 0.001:
            filters.append(f"setpts=PTS/{speed:.8f}")
            if self.video_retime_fps_mode.lower() == "cfr":
                filters.append(f"fps={max(1.0, float(self.source_fps)):.6f}")
        if self.subtitle_path is not None:
            filters.append(f"subtitles={self.subtitle_path.as_posix()}")
        return filters

    def _audio_filters(self, *, force_format: bool = False) -> list[str]:
        filters = [
            f"aresample={int(self.audio_sample_rate_hz)}",
            f"aformat=sample_rates={int(self.audio_sample_rate_hz)}:channel_layouts={'stereo' if int(self.audio_channels) == 2 else 'mono'}",
        ]
        if self.final_smooth:
            filters.append(
                f"acompressor=threshold={float(self.final_smooth_threshold_db):g}dB:"
                f"ratio={float(self.final_smooth_ratio):g}:"
                f"attack={float(self.final_smooth_attack_ms):g}:"
                f"release={float(self.final_smooth_release_ms):g}:"
                f"makeup={float(self.final_smooth_makeup_db):g}"
            )
        if self.final_loudnorm:
            filters.append(
                f"loudnorm=I={float(self.final_loudnorm_i):g}:"
                f"TP={float(self.final_loudnorm_tp):g}:"
                f"LRA={float(self.final_loudnorm_lra):g}:print_format=none"
            )
        return filters if force_format or self.final_loudnorm or self.final_smooth else []


def build_audio_extract_plan(
    source_video: str | Path,
    output_audio: str | Path,
    *,
    sample_rate_hz: int = 44100,
    channels: int = 2,
    audio_codec: str = "",
    audio_bitrate: str = "",
    output_format: str = "",
    metadata: Mapping[str, str] | None = None,
) -> AudioExtractPlan:
    return AudioExtractPlan(
        Path(source_video),
        Path(output_audio),
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        audio_codec=audio_codec,
        audio_bitrate=audio_bitrate,
        output_format=output_format,
        metadata=dict(metadata or {}),
    )


def build_compose_video_plan(
    source_video: str | Path,
    dub_audio: str | Path,
    output_video: str | Path,
    subtitle_path: str | Path | None = None,
    *,
    background_audio: str | Path | None = None,
    video_encoder: str = "",
    audio_bitrate: str = "192k",
    audio_sample_rate_hz: int = 48000,
    audio_channels: int = 2,
    video_speed: float = 1.0,
    video_retime_fps_mode: str = "cfr",
    source_fps: float = 30.0,
    final_loudnorm: bool = False,
    final_loudnorm_i: float = -16.0,
    final_loudnorm_tp: float = -1.5,
    final_loudnorm_lra: float = 4.5,
    final_smooth: bool = False,
    final_smooth_threshold_db: float = -30.0,
    final_smooth_ratio: float = 4.0,
    final_smooth_attack_ms: float = 6.0,
    final_smooth_release_ms: float = 550.0,
    final_smooth_makeup_db: float = 5.5,
    background_audio_speed: float = 1.0,
    background_mix_duration: str = "longest",
) -> ComposeVideoPlan:
    effective_background_audio_speed = background_audio_speed
    if background_audio and abs(float(background_audio_speed) - 1.0) <= 0.001 and abs(float(video_speed) - 1.0) > 0.001:
        effective_background_audio_speed = video_speed
    return ComposeVideoPlan(
        source_video=Path(source_video),
        dub_audio=Path(dub_audio),
        output_video=Path(output_video),
        subtitle_path=Path(subtitle_path) if subtitle_path else None,
        background_audio=Path(background_audio) if background_audio else None,
        video_encoder=video_encoder,
        audio_bitrate=audio_bitrate,
        audio_sample_rate_hz=audio_sample_rate_hz,
        audio_channels=audio_channels,
        video_speed=video_speed,
        video_retime_fps_mode=video_retime_fps_mode,
        source_fps=source_fps,
        final_loudnorm=final_loudnorm,
        final_loudnorm_i=final_loudnorm_i,
        final_loudnorm_tp=final_loudnorm_tp,
        final_loudnorm_lra=final_loudnorm_lra,
        final_smooth=final_smooth,
        final_smooth_threshold_db=final_smooth_threshold_db,
        final_smooth_ratio=final_smooth_ratio,
        final_smooth_attack_ms=final_smooth_attack_ms,
        final_smooth_release_ms=final_smooth_release_ms,
        final_smooth_makeup_db=final_smooth_makeup_db,
        background_audio_speed=effective_background_audio_speed,
        background_mix_duration=background_mix_duration,
    )


def _atempo_filters(speed: float) -> list[str]:
    factors: list[float] = []
    remaining = float(speed)
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    factors.append(remaining)
    return [f"atempo={factor:.6f}" for factor in factors]
