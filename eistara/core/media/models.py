from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class MediaProbeError(RuntimeError):
    """Raised when media metadata cannot be read."""


@dataclass(frozen=True, slots=True)
class VideoStreamInfo:
    codec: str = ""
    width: int | None = None
    height: int | None = None
    duration_sec: float | None = None
    frame_rate: float | None = None


@dataclass(frozen=True, slots=True)
class AudioStreamInfo:
    codec: str = ""
    channels: int | None = None
    sample_rate_hz: int | None = None
    duration_sec: float | None = None


@dataclass(frozen=True, slots=True)
class MediaInfo:
    path: Path
    duration_sec: float | None = None
    format_name: str = ""
    bit_rate: int | None = None
    video: VideoStreamInfo | None = None
    audio: AudioStreamInfo | None = None

    @property
    def has_video(self) -> bool:
        return self.video is not None

    @property
    def has_audio(self) -> bool:
        return self.audio is not None

    @classmethod
    def from_ffprobe(cls, path: str | Path, data: dict[str, Any]) -> "MediaInfo":
        streams = list(data.get("streams") or [])
        format_data = dict(data.get("format") or {})
        video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
        audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
        return cls(
            path=Path(path),
            duration_sec=_optional_float(format_data.get("duration")),
            format_name=str(format_data.get("format_name") or ""),
            bit_rate=_optional_int(format_data.get("bit_rate")),
            video=_video_from_stream(video) if video else None,
            audio=_audio_from_stream(audio) if audio else None,
        )


def _video_from_stream(stream: dict[str, Any]) -> VideoStreamInfo:
    return VideoStreamInfo(
        codec=str(stream.get("codec_name") or ""),
        width=_optional_int(stream.get("width")),
        height=_optional_int(stream.get("height")),
        duration_sec=_optional_float(stream.get("duration")),
        frame_rate=_parse_frame_rate(str(stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "")),
    )


def _audio_from_stream(stream: dict[str, Any]) -> AudioStreamInfo:
    return AudioStreamInfo(
        codec=str(stream.get("codec_name") or ""),
        channels=_optional_int(stream.get("channels")),
        sample_rate_hz=_optional_int(stream.get("sample_rate")),
        duration_sec=_optional_float(stream.get("duration")),
    )


def _optional_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _parse_frame_rate(value: str) -> float | None:
    if not value or value == "0/0":
        return None
    if "/" not in value:
        return _optional_float(value)
    numerator, denominator = value.split("/", 1)
    top = _optional_float(numerator)
    bottom = _optional_float(denominator)
    if not top or not bottom:
        return None
    return top / bottom
