from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import DEFAULT_PUBLISH_TIMELINE_MODE

from eistara.core.translation import TranslationSettings
from eistara.core.tts import TtsSettings


STAGE_NAMES = ("download", "transcribe", "translate", "tts_prepare", "tts", "audio_mix", "compose")
DEFAULT_ALLOWED_VIDEO_FORMATS = ("mp4", "mov", "avi", "mkv", "flv", "wmv", "webm")
DEFAULT_ALLOWED_AUDIO_FORMATS = ("wav", "mp3", "flac", "m4a")
DEFAULT_LANGUAGE_SPLIT_WITH_SPACE = ("en", "es", "fr", "de", "it", "ru")
DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE = ("zh", "ja")


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key) or {}
    return dict(value) if isinstance(value, dict) else {}


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _as_int(value: Any, default: int) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    return str(value)


def _as_list(value: Any, default: list[str] | tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return list(default)


def _merged(*sections: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in sections:
        result.update(dict(section))
    return result


def _merge_source_provider_config(source: dict[str, Any], youtube: "YouTubeConfig") -> dict[str, Any]:
    result = dict(source)
    for key, value in youtube.to_provider_config().items():
        current = result.get(key)
        if key in {"auto_update_ytdlp", "write_thumbnail"}:
            if current in {None, False}:
                result[key] = value
            continue
        if key == "socket_timeout":
            if current in (None, "", 20, 20.0):
                result[key] = value
            continue
        if key == "retries":
            if current in (None, "", 3):
                result[key] = value
            continue
        if key == "cookies_from_browser":
            if current in (None, "", "auto", "default", "default_browser"):
                result[key] = value
            continue
        if current in (None, "", [], {}):
            result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ApiConfig:
    key: str = ""
    base_url: str = ""
    model: str = ""
    llm_interface: str = "chat_completions"
    llm_stream: bool = False
    llm_support_json: bool = True
    timeout_sec: float = 300.0
    user_agent: str = "curl/8.19.0"
    trust_env_proxy: bool = True
    proxy_url: str = ""
    max_retries: int = 6
    retry_base_delay_sec: float = 4.0
    retry_max_delay_sec: float = 60.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApiConfig":
        return cls(
            key=_as_str(data.get("key")),
            base_url=_as_str(data.get("base_url")),
            model=_as_str(data.get("model")),
            llm_interface=_as_str(data.get("llm_interface"), "chat_completions").strip().lower().replace("-", "_"),
            llm_stream=_as_bool(data.get("llm_stream"), False),
            llm_support_json=_as_bool(data.get("llm_support_json"), True),
            timeout_sec=_as_float(data.get("timeout_sec"), 300.0),
            user_agent=_as_str(data.get("user_agent"), "curl/8.19.0"),
            trust_env_proxy=_as_bool(data.get("trust_env_proxy"), True),
            proxy_url=_as_str(data.get("proxy_url")),
            max_retries=_as_int(data.get("max_retries"), 6),
            retry_base_delay_sec=_as_float(data.get("retry_base_delay_sec"), 4.0),
            retry_max_delay_sec=_as_float(data.get("retry_max_delay_sec"), 60.0),
        )


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    display_language: str = "zh-CN"
    max_workers: int = 4
    llm_max_workers: int = 4
    tts_max_workers: int = 1
    summary_length: int = 8000
    model_dir: Path = Path("./_model_cache")
    allowed_video_formats: tuple[str, ...] = DEFAULT_ALLOWED_VIDEO_FORMATS
    allowed_audio_formats: tuple[str, ...] = DEFAULT_ALLOWED_AUDIO_FORMATS
    spacy_model_map: dict[str, str] = field(default_factory=dict)
    language_split_with_space: tuple[str, ...] = DEFAULT_LANGUAGE_SPLIT_WITH_SPACE
    language_split_without_space: tuple[str, ...] = DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        return cls(
            display_language=_as_str(data.get("display_language"), "zh-CN"),
            max_workers=_as_int(data.get("max_workers"), 4),
            llm_max_workers=_as_int(data.get("llm_max_workers"), 4),
            tts_max_workers=_as_int(data.get("tts_max_workers"), 1),
            summary_length=_as_int(data.get("summary_length"), 8000),
            model_dir=Path(_as_str(data.get("model_dir"), "./_model_cache")),
            allowed_video_formats=tuple(_as_list(data.get("allowed_video_formats"), DEFAULT_ALLOWED_VIDEO_FORMATS)),
            allowed_audio_formats=tuple(_as_list(data.get("allowed_audio_formats"), DEFAULT_ALLOWED_AUDIO_FORMATS)),
            spacy_model_map={str(key): str(value) for key, value in _section(data, "spacy_model_map").items()},
            language_split_with_space=tuple(_as_list(data.get("language_split_with_space"), DEFAULT_LANGUAGE_SPLIT_WITH_SPACE)),
            language_split_without_space=tuple(_as_list(data.get("language_split_without_space"), DEFAULT_LANGUAGE_SPLIT_WITHOUT_SPACE)),
        )


@dataclass(frozen=True, slots=True)
class BatchConfig:
    jobs_dir: Path = Path("jobs")
    max_active_jobs: int = 10
    download_workers: int = 3
    transcribe_workers: int = 1
    translate_workers: int = 1
    tts_prepare_workers: int = 1
    tts_workers: int = 1
    audio_mix_workers: int = 1
    compose_workers: int = 1
    poll_interval_sec: int = 3
    max_stage_retries: int = 1
    stage_idle_timeout_sec: int = 0
    download_idle_timeout_sec: int = 1800
    transcribe_idle_timeout_sec: int = 7200
    translate_idle_timeout_sec: int = 1800
    tts_prepare_idle_timeout_sec: int = 900
    tts_idle_timeout_sec: int = 1800
    audio_mix_idle_timeout_sec: int = 1800
    compose_idle_timeout_sec: int = 7200
    dependency_probe: bool = True
    dependency_probe_ttl_sec: int = 30
    auto_requeue_failed: bool = True
    failed_cooldown_sec: int = 300
    max_auto_requeues: int = 2

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BatchConfig":
        dub_workers = data.get("dub_workers")
        return cls(
            jobs_dir=Path(_as_str(data.get("jobs_dir"), "jobs")),
            max_active_jobs=_as_int(data.get("max_active_jobs"), 10),
            download_workers=_as_int(data.get("download_workers"), 3),
            transcribe_workers=_as_int(data.get("transcribe_workers"), 1),
            translate_workers=_as_int(data.get("translate_workers"), 1),
            tts_prepare_workers=_as_int(data.get("tts_prepare_workers"), 1),
            tts_workers=_as_int(data.get("tts_workers", dub_workers), 1),
            audio_mix_workers=_as_int(data.get("audio_mix_workers"), 1),
            compose_workers=_as_int(data.get("compose_workers"), 1),
            poll_interval_sec=_as_int(data.get("poll_interval_sec"), 3),
            max_stage_retries=_as_int(data.get("max_stage_retries"), 1),
            stage_idle_timeout_sec=_as_int(data.get("stage_idle_timeout_sec"), 0),
            download_idle_timeout_sec=_as_int(data.get("download_idle_timeout_sec"), 1800),
            transcribe_idle_timeout_sec=_as_int(data.get("transcribe_idle_timeout_sec"), 7200),
            translate_idle_timeout_sec=_as_int(data.get("translate_idle_timeout_sec"), 1800),
            tts_prepare_idle_timeout_sec=_as_int(data.get("tts_prepare_idle_timeout_sec"), 900),
            tts_idle_timeout_sec=_as_int(data.get("tts_idle_timeout_sec"), 1800),
            audio_mix_idle_timeout_sec=_as_int(data.get("audio_mix_idle_timeout_sec"), 1800),
            compose_idle_timeout_sec=_as_int(data.get("compose_idle_timeout_sec"), 7200),
            dependency_probe=_as_bool(data.get("dependency_probe"), True),
            dependency_probe_ttl_sec=_as_int(data.get("dependency_probe_ttl_sec"), 30),
            auto_requeue_failed=_as_bool(data.get("auto_requeue_failed"), True),
            failed_cooldown_sec=_as_int(data.get("failed_cooldown_sec"), 300),
            max_auto_requeues=_as_int(data.get("max_auto_requeues"), 2),
        )

    def stage_worker_limits(self) -> dict[str, int]:
        return {
            "download": self.download_workers,
            "transcribe": self.transcribe_workers,
            "translate": self.translate_workers,
            "tts_prepare": self.tts_prepare_workers,
            "tts": self.tts_workers,
            "audio_mix": self.audio_mix_workers,
            "compose": self.compose_workers,
        }

    def stage_idle_timeouts(self) -> dict[str, int]:
        return {
            "download": self.download_idle_timeout_sec,
            "transcribe": self.transcribe_idle_timeout_sec,
            "translate": self.translate_idle_timeout_sec,
            "tts_prepare": self.tts_prepare_idle_timeout_sec,
            "tts": self.tts_idle_timeout_sec,
            "audio_mix": self.audio_mix_idle_timeout_sec,
            "compose": self.compose_idle_timeout_sec,
        }


@dataclass(frozen=True, slots=True)
class AsrConfig:
    provider: str = "local"
    model: str = "large-v3"
    language: str = "en"
    provider_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sections(cls, asr: dict[str, Any], whisper: dict[str, Any]) -> "AsrConfig":
        provider_config = _merged(whisper, asr)
        model = _as_str(asr.get("model"), "large-v3")
        language = _as_str(asr.get("language"), "en")
        whisper_model = _as_str(whisper.get("model"))
        whisper_language = _as_str(whisper.get("language"))
        runtime = _as_str(whisper.get("runtime"), "local").strip().lower()
        runtime = runtime if runtime in {"local", "cloud", "elevenlabs"} else "local"
        if whisper_model and model in {"", "base", "large-v3"}:
            model = whisper_model
        if whisper_language and language in {"", "en"}:
            language = whisper_language
        provider_config.pop("provider", None)
        provider_config["runtime"] = runtime
        provider_config["model"] = model
        provider_config["language"] = language
        return cls(
            provider=runtime,
            model=model,
            language=language,
            provider_config=provider_config,
        )


@dataclass(frozen=True, slots=True)
class YouTubeConfig:
    cookies_path: str = ""
    cookies_from_browser: str = ""
    cookies_browser_profile: str = ""
    auto_update_ytdlp: bool = False
    write_thumbnail: bool = False
    socket_timeout: float = 20.0
    retries: int = 3

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "YouTubeConfig":
        return cls(
            cookies_path=_as_str(data.get("cookies_path")),
            cookies_from_browser=_as_str(data.get("cookies_from_browser")),
            cookies_browser_profile=_as_str(data.get("cookies_browser_profile")),
            auto_update_ytdlp=_as_bool(data.get("auto_update_ytdlp"), False),
            write_thumbnail=_as_bool(data.get("write_thumbnail"), False),
            socket_timeout=_as_float(data.get("socket_timeout"), 20.0),
            retries=_as_int(data.get("retries"), 3),
        )

    def to_provider_config(self) -> dict[str, Any]:
        return {
            "cookies_path": self.cookies_path,
            "cookies_from_browser": self.cookies_from_browser,
            "cookies_browser_profile": self.cookies_browser_profile,
            "auto_update_ytdlp": self.auto_update_ytdlp,
            "write_thumbnail": self.write_thumbnail,
            "socket_timeout": self.socket_timeout,
            "retries": self.retries,
        }


@dataclass(frozen=True, slots=True)
class SourceConfig:
    url_provider: str = "yt-dlp"
    output_filename: str = "source_video.mp4"
    yt_dlp_path: str = "yt-dlp"
    resolution: str = "1080"
    provider_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sections(
        cls,
        source: dict[str, Any],
        youtube: YouTubeConfig,
        runtime: RuntimeConfig,
        ytb_resolution: Any,
    ) -> "SourceConfig":
        source_resolution = _as_str(source.get("resolution"), "1080")
        top_level_resolution = _as_str(ytb_resolution)
        resolution = top_level_resolution if top_level_resolution and source_resolution in {"", "1080"} else source_resolution
        provider_config = _merge_source_provider_config(source, youtube)
        provider_config.update(
            {
                "resolution": resolution,
                "ytb_resolution": resolution,
                "allowed_video_formats": list(runtime.allowed_video_formats),
                "allowed_audio_formats": list(runtime.allowed_audio_formats),
            }
        )
        return cls(
            url_provider=_as_str(source.get("url_provider"), "yt-dlp"),
            output_filename=_as_str(source.get("output_filename"), "source_video.mp4"),
            yt_dlp_path=_as_str(source.get("yt_dlp_path"), "yt-dlp"),
            resolution=resolution,
            provider_config=provider_config,
        )


@dataclass(frozen=True, slots=True)
class MediaConfig:
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    ffmpeg_gpu: bool = False

    @classmethod
    def from_sections(cls, media: dict[str, Any], top_level_ffmpeg_gpu: Any) -> "MediaConfig":
        ffmpeg_gpu = media.get("ffmpeg_gpu")
        if top_level_ffmpeg_gpu is not None and ffmpeg_gpu in (None, False):
            ffmpeg_gpu = top_level_ffmpeg_gpu
        return cls(
            ffmpeg_path=_as_str(media.get("ffmpeg_path"), "ffmpeg"),
            ffprobe_path=_as_str(media.get("ffprobe_path"), "ffprobe"),
            ffmpeg_gpu=_as_bool(ffmpeg_gpu, False),
        )


@dataclass(frozen=True, slots=True)
class RenderConfig:
    render_audio: bool = True
    render_video: bool = True
    burn_subtitles: bool = False

    @classmethod
    def from_sections(cls, render: dict[str, Any], top_level_burn_subtitles: Any) -> "RenderConfig":
        burn_subtitles = render.get("burn_subtitles")
        if top_level_burn_subtitles is not None and burn_subtitles in (None, False):
            burn_subtitles = top_level_burn_subtitles
        return cls(
            render_audio=_as_bool(render.get("render_audio"), True),
            render_video=_as_bool(render.get("render_video"), True),
            burn_subtitles=_as_bool(burn_subtitles, False),
        )


@dataclass(frozen=True, slots=True)
class DemucsConfig:
    enabled: bool = True
    segment_minutes: float = 30.0

    @classmethod
    def from_sections(cls, enabled: Any, segment_minutes: Any) -> "DemucsConfig":
        section = dict(enabled) if isinstance(enabled, dict) else {}
        enabled_value = section.get("enabled") if section else enabled
        segment_minutes_value = section.get("segment_minutes", segment_minutes)
        return cls(
            enabled=_as_bool(enabled_value, True),
            segment_minutes=_as_float(segment_minutes_value, 30.0),
        )


@dataclass(frozen=True, slots=True)
class SubtitleConfig:
    display_max_chars_per_line: int = 20
    display_source_max_chars_per_line: int = 42
    display_max_lines: int = 1

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SubtitleConfig":
        return cls(
            display_max_chars_per_line=_as_int(data.get("display_max_chars_per_line"), 20),
            display_source_max_chars_per_line=_as_int(data.get("display_source_max_chars_per_line"), 42),
            display_max_lines=_as_int(data.get("display_max_lines"), 1),
        )


@dataclass(frozen=True, slots=True)
class DubAudioConfig:
    bitrate: str = "192k"
    sample_rate: int = 24000
    publish_timeline_mode: str = DEFAULT_PUBLISH_TIMELINE_MODE
    publish_lead_in_ms: int = 300
    publish_line_gap_ms: int = 180
    publish_row_gap_ms: int = 260
    publish_source_gap_scale: float = 0.45
    publish_min_source_gap_sec: float = 0.12
    publish_max_source_gap_sec: float = 6.0
    publish_preserve_short_source_windows: bool = False
    publish_source_window_stretch_max: float = 1.10
    publish_source_window_borrow_enabled: bool = True
    publish_source_window_borrow_max_sec: float = 0.60
    publish_source_window_borrow_max_ratio: float = 0.50
    publish_source_window_borrow_min_seam_sec: float = 0.12
    publish_source_window_retime_tier2_enabled: bool = False
    publish_tail_pad_ms: int = 500
    publish_global_audio_speed: bool = True
    publish_target_video_speed_min: float = 0.90
    publish_max_audio_speed: float = 1.10
    publish_short_video_speed_max: float = 1.05
    publish_short_video_speed_hard_max: float = 1.08
    video_retime: bool = True
    video_retime_fps_mode: str = "cfr"
    video_speed_warn_min: float = 0.75
    video_speed_warn_max: float = 1.35
    background_ducking: bool = True
    background_bed_mode: str = "separated"
    background_duck_volume: float = 0.35
    background_duck_high_coverage_threshold: float = 0.85
    background_duck_high_coverage_volume: float = 0.35
    background_duck_padding_ms: int = 120
    background_duck_merge_gap_ms: int = 700
    background_duck_transition_ms: int = 600
    background_duck_filter: bool = True
    background_duck_lowpass_hz: int = 4200
    background_duck_adaptive: bool = True
    background_duck_target_under_voice_db: float = 16.0
    background_duck_high_coverage_under_voice_db: float = 14.0
    background_duck_max_makeup_db: float = 12.0
    background_duck_wideband_when_adaptive: bool = True
    source_bed_duck_volume: float = 0.12
    source_bed_lowpass_hz: int = 3600
    final_loudnorm: bool = True
    final_loudnorm_i: float = -16.0
    final_loudnorm_tp: float = -1.5
    final_loudnorm_lra: float = 4.5
    final_smooth: bool = False
    final_smooth_preset: str = "broadcast"
    final_smooth_threshold_db: float = -30.0
    final_smooth_ratio: float = 4.0
    final_smooth_attack_ms: float = 6.0
    final_smooth_release_ms: float = 550.0
    final_smooth_makeup_db: float = 5.5
    final_audio_sample_rate: int = 48000
    final_audio_channels: int = 2
    tts_segment_fade_in_ms: int = 5
    tts_segment_fade_out_ms: int = 220
    tts_segment_tail_pad_ms: int = 220
    tts_segment_tail_pad_counts_in_timeline: bool = False
    tts_segment_tail_cleanup: bool = True
    tts_segment_tail_cleanup_ms: int = 420
    tts_segment_tail_cleanup_lowpass_hz: int = 3600

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DubAudioConfig":
        return cls(
            bitrate=_as_str(data.get("bitrate"), "192k"),
            sample_rate=_as_int(data.get("sample_rate"), 24000),
            publish_timeline_mode=_as_str(data.get("publish_timeline_mode"), DEFAULT_PUBLISH_TIMELINE_MODE),
            publish_lead_in_ms=_as_int(data.get("publish_lead_in_ms"), 300),
            publish_line_gap_ms=_as_int(data.get("publish_line_gap_ms"), 180),
            publish_row_gap_ms=_as_int(data.get("publish_row_gap_ms"), 260),
            publish_source_gap_scale=_as_float(data.get("publish_source_gap_scale"), 0.45),
            publish_min_source_gap_sec=_as_float(data.get("publish_min_source_gap_sec"), 0.12),
            publish_max_source_gap_sec=_as_float(data.get("publish_max_source_gap_sec"), 6.0),
            publish_preserve_short_source_windows=_as_bool(data.get("publish_preserve_short_source_windows"), False),
            publish_source_window_stretch_max=_as_float(data.get("publish_source_window_stretch_max"), 1.10),
            publish_source_window_borrow_enabled=_as_bool(data.get("publish_source_window_borrow_enabled"), True),
            publish_source_window_borrow_max_sec=_as_float(data.get("publish_source_window_borrow_max_sec"), 0.60),
            publish_source_window_borrow_max_ratio=_as_float(data.get("publish_source_window_borrow_max_ratio"), 0.50),
            publish_source_window_borrow_min_seam_sec=_as_float(data.get("publish_source_window_borrow_min_seam_sec"), 0.12),
            publish_source_window_retime_tier2_enabled=_as_bool(data.get("publish_source_window_retime_tier2_enabled"), False),
            publish_tail_pad_ms=_as_int(data.get("publish_tail_pad_ms"), 500),
            publish_global_audio_speed=_as_bool(data.get("publish_global_audio_speed"), True),
            publish_target_video_speed_min=_as_float(data.get("publish_target_video_speed_min"), 0.90),
            publish_max_audio_speed=_as_float(data.get("publish_max_audio_speed"), 1.10),
            publish_short_video_speed_max=_as_float(data.get("publish_short_video_speed_max"), 1.05),
            publish_short_video_speed_hard_max=_as_float(data.get("publish_short_video_speed_hard_max"), 1.08),
            video_retime=_as_bool(data.get("video_retime"), True),
            video_retime_fps_mode=_as_str(data.get("video_retime_fps_mode"), "cfr"),
            video_speed_warn_min=_as_float(data.get("video_speed_warn_min"), 0.75),
            video_speed_warn_max=_as_float(data.get("video_speed_warn_max"), 1.35),
            background_ducking=_as_bool(data.get("background_ducking"), True),
            background_bed_mode=_as_str(data.get("background_bed_mode"), "separated"),
            background_duck_volume=_as_float(data.get("background_duck_volume"), 0.35),
            background_duck_high_coverage_threshold=_as_float(data.get("background_duck_high_coverage_threshold"), 0.85),
            background_duck_high_coverage_volume=_as_float(data.get("background_duck_high_coverage_volume"), 0.35),
            background_duck_padding_ms=_as_int(data.get("background_duck_padding_ms"), 120),
            background_duck_merge_gap_ms=_as_int(data.get("background_duck_merge_gap_ms"), 700),
            background_duck_transition_ms=_as_int(data.get("background_duck_transition_ms"), 600),
            background_duck_filter=_as_bool(data.get("background_duck_filter"), True),
            background_duck_lowpass_hz=_as_int(data.get("background_duck_lowpass_hz"), 4200),
            background_duck_adaptive=_as_bool(data.get("background_duck_adaptive"), True),
            background_duck_target_under_voice_db=_as_float(data.get("background_duck_target_under_voice_db"), 16.0),
            background_duck_high_coverage_under_voice_db=_as_float(data.get("background_duck_high_coverage_under_voice_db"), 14.0),
            background_duck_max_makeup_db=_as_float(data.get("background_duck_max_makeup_db"), 12.0),
            background_duck_wideband_when_adaptive=_as_bool(data.get("background_duck_wideband_when_adaptive"), True),
            source_bed_duck_volume=_as_float(data.get("source_bed_duck_volume"), 0.12),
            source_bed_lowpass_hz=_as_int(data.get("source_bed_lowpass_hz"), 3600),
            final_loudnorm=_as_bool(data.get("final_loudnorm"), True),
            final_loudnorm_i=_as_float(data.get("final_loudnorm_i"), -16.0),
            final_loudnorm_tp=_as_float(data.get("final_loudnorm_tp"), -1.5),
            final_loudnorm_lra=_as_float(data.get("final_loudnorm_lra"), 4.5),
            final_smooth=_as_bool(data.get("final_smooth"), False),
            final_smooth_preset=_as_str(data.get("final_smooth_preset"), "broadcast"),
            final_smooth_threshold_db=_as_float(data.get("final_smooth_threshold_db"), -30.0),
            final_smooth_ratio=_as_float(data.get("final_smooth_ratio"), 4.0),
            final_smooth_attack_ms=_as_float(data.get("final_smooth_attack_ms"), 6.0),
            final_smooth_release_ms=_as_float(data.get("final_smooth_release_ms"), 550.0),
            final_smooth_makeup_db=_as_float(data.get("final_smooth_makeup_db"), 5.5),
            final_audio_sample_rate=_as_int(data.get("final_audio_sample_rate"), 48000),
            final_audio_channels=_as_int(data.get("final_audio_channels"), 2),
            tts_segment_fade_in_ms=_as_int(data.get("tts_segment_fade_in_ms"), 5),
            tts_segment_fade_out_ms=_as_int(data.get("tts_segment_fade_out_ms"), 220),
            tts_segment_tail_pad_ms=_as_int(data.get("tts_segment_tail_pad_ms"), 220),
            tts_segment_tail_pad_counts_in_timeline=_as_bool(data.get("tts_segment_tail_pad_counts_in_timeline"), False),
            tts_segment_tail_cleanup=_as_bool(data.get("tts_segment_tail_cleanup"), True),
            tts_segment_tail_cleanup_ms=_as_int(data.get("tts_segment_tail_cleanup_ms"), 420),
            tts_segment_tail_cleanup_lowpass_hz=_as_int(data.get("tts_segment_tail_cleanup_lowpass_hz"), 3600),
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    api: ApiConfig = ApiConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    batch: BatchConfig = BatchConfig()
    source: SourceConfig = SourceConfig()
    youtube: YouTubeConfig = YouTubeConfig()
    asr: AsrConfig = AsrConfig()
    demucs: DemucsConfig = DemucsConfig()
    subtitle: SubtitleConfig = SubtitleConfig()
    media: MediaConfig = MediaConfig()
    render: RenderConfig = RenderConfig()
    dub_audio: DubAudioConfig = DubAudioConfig()
    target_language: str = "Simplified Chinese"
    translation: dict[str, Any] = field(default_factory=dict)
    tts_method: str = "indextts"
    tts_audio: dict[str, Any] = field(default_factory=dict)
    indextts: dict[str, Any] = field(default_factory=dict)
    custom_tts: dict[str, Any] = field(default_factory=dict)
    sf_fish_tts: dict[str, Any] = field(default_factory=dict)
    openai_tts: dict[str, Any] = field(default_factory=dict)
    azure_tts: dict[str, Any] = field(default_factory=dict)
    fish_tts: dict[str, Any] = field(default_factory=dict)
    sf_cosyvoice2: dict[str, Any] = field(default_factory=dict)
    edge_tts: dict[str, Any] = field(default_factory=dict)
    gpt_sovits: dict[str, Any] = field(default_factory=dict)
    f5tts: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        data = dict(data or {})
        runtime = RuntimeConfig.from_dict(data)
        batch = BatchConfig.from_dict(_section(data, "batch"))
        youtube = YouTubeConfig.from_dict(_section(data, "youtube"))
        source = SourceConfig.from_sections(_section(data, "source"), youtube, runtime, data.get("ytb_resolution"))
        return cls(
            api=ApiConfig.from_dict(_section(data, "api")),
            runtime=runtime,
            batch=batch,
            source=source,
            youtube=youtube,
            asr=AsrConfig.from_sections(_section(data, "asr"), _section(data, "whisper")),
            demucs=DemucsConfig.from_sections(data.get("demucs"), data.get("demucs_segment_minutes")),
            subtitle=SubtitleConfig.from_dict(_section(data, "subtitle")),
            media=MediaConfig.from_sections(_section(data, "media"), data.get("ffmpeg_gpu")),
            render=RenderConfig.from_sections(_section(data, "render"), data.get("burn_subtitles")),
            dub_audio=DubAudioConfig.from_dict(_section(data, "dub_audio")),
            target_language=_as_str(data.get("target_language"), "Simplified Chinese"),
            translation=dict(data.get("translation") or {}),
            tts_method=_as_str(data.get("tts_method"), "indextts"),
            tts_audio=dict(data.get("tts_audio") or {}),
            indextts=dict(data.get("indextts") or {}),
            custom_tts=dict(data.get("custom_tts") or {}),
            sf_fish_tts=dict(data.get("sf_fish_tts") or {}),
            openai_tts=dict(data.get("openai_tts") or {}),
            azure_tts=dict(data.get("azure_tts") or {}),
            fish_tts=dict(data.get("fish_tts") or {}),
            sf_cosyvoice2=dict(data.get("sf_cosyvoice2") or {}),
            edge_tts=dict(data.get("edge_tts") or {}),
            gpt_sovits=dict(data.get("gpt_sovits") or {}),
            f5tts=dict(data.get("f5tts") or {}),
            raw=data,
        )

    def asr_settings(self):
        from eistara.core.asr import AsrSettings

        provider_config = dict(self.asr.provider_config)
        provider_config.update(self.youtube.to_provider_config())
        provider_config["demucs"] = self.demucs.enabled
        provider_config["demucs_segment_minutes"] = self.demucs.segment_minutes
        provider_config["model_dir"] = str(self.runtime.model_dir)
        provider_config["spacy_model_map"] = dict(self.runtime.spacy_model_map)
        provider_config["language_split_with_space"] = list(self.runtime.language_split_with_space)
        provider_config["language_split_without_space"] = list(self.runtime.language_split_without_space)
        return AsrSettings(
            language=self.asr.language or None,
            model=self.asr.model,
            provider_config=provider_config,
        )

    def source_settings(self):
        from eistara.core.source import SourceSettings

        return SourceSettings(
            output_filename=self.source.output_filename,
            provider_config=dict(self.source.provider_config),
        )

    def translation_settings(self, source_language: str = "source language") -> TranslationSettings:
        return TranslationSettings(
            source_language=source_language,
            target_language=self.target_language,
            max_batch_lines=_as_int(self.translation.get("publish_fast_chunk_lines"), 20),
            max_batch_chars=_as_int(self.translation.get("publish_fast_chunk_chars"), 3000),
            use_summary=_as_bool(self.translation.get("publish_fast_use_summary"), True),
            summary_length=self.runtime.summary_length,
            enforce_latin=_as_bool(self.translation.get("enforce_latin"), True),
            localization_chars_per_sec=_as_float(self.translation.get("localization_chars_per_sec"), 4.2),
            localization_spoken_cost_per_sec=_as_float(self.translation.get("localization_spoken_cost_per_sec"), 3.6),
            localization_max_audio_speed=_as_float(self.translation.get("localization_max_audio_speed"), 1.10),
            localization_seam_gap_sec=_as_float(self.translation.get("localization_seam_gap_sec"), 0.12),
            localization_max_window_gap_sec=_as_float(self.translation.get("localization_max_window_gap_sec"), 6.0),
            raw_config=dict(self.raw),
        )

    def tts_backend_config(self, method: str | None = None) -> dict[str, Any]:
        selected = (method or self.tts_method).strip()
        by_method = {
            "indextts": self.indextts,
            "custom_tts": self.custom_tts,
            "sf_fish_tts": self.sf_fish_tts,
            "openai_tts": self.openai_tts,
            "azure_tts": self.azure_tts,
            "fish_tts": self.fish_tts,
            "sf_cosyvoice2": self.sf_cosyvoice2,
            "edge_tts": self.edge_tts,
            "gpt_sovits": self.gpt_sovits,
            "f5tts": self.f5tts,
        }
        return dict(by_method.get(selected, {}))

    def tts_settings(self) -> TtsSettings:
        provider_config = self.tts_backend_config(self.tts_method)
        audio_config = dict(self.tts_audio)
        audio_config["ffmpeg_path"] = self.media.ffmpeg_path
        return TtsSettings(
            method=self.tts_method,
            provider_config=provider_config,
            audio_config=audio_config,
            max_retries=_as_int(provider_config.get("max_retries"), 3),
            service_backoff_base_sec=_as_float(provider_config.get("service_backoff_base_sec"), 2.0),
        )
