from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eistara.adapters.asr import DemucsVocalSeparationProvider, WhisperRuntimeAsrProvider
from eistara.adapters.llm import CodexResponsesLlmClient, OpenAICompatibleLlmClient, OpenAICompatibleSettings, RequestsHttpTransport
from eistara.adapters.media import FfmpegDubbingRenderer, FfmpegMediaProvider
from eistara.adapters.source import YtDlpSourceProvider
from eistara.adapters.tts import IndexTtsProvider
from eistara.config import AppConfig, ConfigLoader
from eistara.core.asr import AsrProvider, TranscribeStageRunner
from eistara.core.dubbing import AudioMixPlanStageRunner, ComposePlanStageRunner, DubbingRenderService
from eistara.core.jobs import STAGE_ORDER, StageName
from eistara.core.media import MediaProvider
from eistara.core.pipeline import StageContext, StageResult, noop_runners
from eistara.core.scheduler import SchedulerDependencyProbe, SchedulerPolicy, SchedulerProcessSupervisor, SchedulerRecoveryPolicy, SchedulerService
from eistara.core.source import SourceStageRunner
from eistara.core.source import SourceProvider
from eistara.core.timeline import TimelinePolicy, TimelinePreparationService
from eistara.core.translation import LlmClient, PublishTranslationStageRunner
from eistara.core.tts import TtsPrepareStageRunner, TtsProvider, TtsStageRunner

from .health import default_llm_chat_probe


@dataclass(frozen=True, slots=True)
class ExistingArtifactStageRunner:
    stage: StageName
    task_key: str
    output_key: str
    missing_warning: str

    def run(self, context: StageContext) -> StageResult:
        value = context.task.get(self.task_key) or context.artifacts.get(self.task_key)
        if not value:
            return StageResult(status="skipped", skipped=True, warnings=[self.missing_warning])
        return StageResult(outputs={self.output_key: str(value)})


@dataclass(frozen=True, slots=True)
class ExistingCollectionStageRunner:
    stage: StageName
    task_key: str
    output_key: str
    missing_warning: str

    def run(self, context: StageContext) -> StageResult:
        value = context.task.get(self.task_key) or context.artifacts.get(self.task_key)
        if not value:
            return StageResult(status="skipped", skipped=True, warnings=[self.missing_warning])
        count = len(value) if hasattr(value, "__len__") else 1
        return StageResult(outputs={self.output_key: value, f"{self.output_key}_count": count})


@dataclass(frozen=True, slots=True)
class RuntimeProviders:
    source_url_provider: SourceProvider | None = None
    media_provider: MediaProvider | None = None
    asr_provider: AsrProvider | None = None
    llm_client: LlmClient | None = None
    tts_provider: TtsProvider | None = None


def build_scheduler(
    jobs_dir: str | Path,
    *,
    preset: str = "noop",
    config: AppConfig | None = None,
    providers: RuntimeProviders | None = None,
    render_audio: bool = False,
    render_video: bool = False,
    max_stage_retries: int | None = None,
) -> SchedulerService:
    config = config or ConfigLoader().load()
    dependencies = (
        SchedulerDependencyProbe.from_config(config, llm_chat_probe=default_llm_chat_probe)
        if _preset_uses_external_services(preset)
        else SchedulerDependencyProbe(enabled=False, config=config)
    )
    service = SchedulerService(
        Path(jobs_dir),
        policy=SchedulerPolicy.from_batch_config(config.batch),
        dependencies=dependencies,
        recovery=SchedulerRecoveryPolicy.from_batch_config(config.batch),
    )
    service.max_stage_retries = config.batch.max_stage_retries if max_stage_retries is None else max_stage_retries
    for runner in build_runners(preset, config=config, providers=providers, render_audio=render_audio, render_video=render_video):
        service.register(runner)
    return service


def build_process_supervisor(
    jobs_dir: str | Path,
    *,
    preset: str = "noop",
    config: AppConfig | None = None,
    config_path: str | Path | None = None,
    render_audio: bool = False,
    render_video: bool = False,
    max_stage_retries: int | None = None,
) -> SchedulerProcessSupervisor:
    service = build_scheduler(
        jobs_dir,
        preset=preset,
        config=config,
        render_audio=render_audio,
        render_video=render_video,
        max_stage_retries=max_stage_retries,
    )
    return SchedulerProcessSupervisor(
        service,
        preset=preset,
        config_path=config_path,
        render_audio=render_audio,
        render_video=render_video,
    )


def _preset_uses_external_services(preset: str) -> bool:
    return preset == "production"


def build_runners(
    preset: str,
    *,
    config: AppConfig | None = None,
    providers: RuntimeProviders | None = None,
    render_audio: bool = False,
    render_video: bool = False,
):
    config = config or ConfigLoader().load()
    providers = providers or RuntimeProviders()
    if preset == "noop":
        return noop_runners(STAGE_ORDER)
    if preset == "plan":
        return _plan_runners(config, render_audio=render_audio, render_video=render_video)
    if preset == "production":
        renderer = _renderer(config, render_audio=render_audio, render_video=render_video)
        media_provider = providers.media_provider or _media_provider(config)
        dubbing_service = _dubbing_service(config)
        return [
            SourceStageRunner(url_provider=providers.source_url_provider or _source_url_provider(config), settings=config.source_settings()),
            TranscribeStageRunner(
                providers.asr_provider or _asr_provider(config),
                media_provider,
                config.asr_settings(),
                vocal_separation_provider=_vocal_separation_provider(config),
            ),
            PublishTranslationStageRunner(providers.llm_client or _llm_client(config), config.translation_settings()),
            TtsPrepareStageRunner(audio_config=config.tts_settings().audio_config),
            TtsStageRunner(providers.tts_provider or _tts_provider(config), config.tts_settings(), media_probe=media_provider),
            AudioMixPlanStageRunner(
                service=dubbing_service,
                renderer=renderer,
                timeline_preparation=TimelinePreparationService(media_provider),
                timeline_policy=_timeline_policy(config),
                render_audio=render_audio or config.render.render_audio,
            ),
            ComposePlanStageRunner(
                service=dubbing_service,
                renderer=renderer,
                render_video=render_video or config.render.render_video,
                burn_subtitles=config.render.burn_subtitles,
                video_encoder="h264_nvenc" if config.media.ffmpeg_gpu else "",
                audio_bitrate=config.dub_audio.bitrate,
                audio_sample_rate_hz=config.dub_audio.final_audio_sample_rate,
                audio_channels=config.dub_audio.final_audio_channels,
                video_retime=config.dub_audio.video_retime,
                video_retime_fps_mode=config.dub_audio.video_retime_fps_mode,
                video_speed_warn_min=config.dub_audio.video_speed_warn_min,
                video_speed_warn_max=config.dub_audio.video_speed_warn_max,
                final_loudnorm=config.dub_audio.final_loudnorm,
                final_loudnorm_i=config.dub_audio.final_loudnorm_i,
                final_loudnorm_tp=config.dub_audio.final_loudnorm_tp,
                final_loudnorm_lra=config.dub_audio.final_loudnorm_lra,
                final_smooth=config.dub_audio.final_smooth,
                final_smooth_threshold_db=config.dub_audio.final_smooth_threshold_db,
                final_smooth_ratio=config.dub_audio.final_smooth_ratio,
                final_smooth_attack_ms=config.dub_audio.final_smooth_attack_ms,
                final_smooth_release_ms=config.dub_audio.final_smooth_release_ms,
                final_smooth_makeup_db=config.dub_audio.final_smooth_makeup_db,
                background_ducking=config.dub_audio.background_ducking,
                background_bed_mode=config.dub_audio.background_bed_mode,
                background_duck_volume=config.dub_audio.background_duck_volume,
                background_duck_high_coverage_threshold=config.dub_audio.background_duck_high_coverage_threshold,
                background_duck_high_coverage_volume=config.dub_audio.background_duck_high_coverage_volume,
                background_duck_padding_ms=config.dub_audio.background_duck_padding_ms,
                background_duck_merge_gap_ms=config.dub_audio.background_duck_merge_gap_ms,
                background_duck_transition_ms=config.dub_audio.background_duck_transition_ms,
                background_duck_filter=config.dub_audio.background_duck_filter,
                background_duck_lowpass_hz=config.dub_audio.background_duck_lowpass_hz,
                background_duck_adaptive=config.dub_audio.background_duck_adaptive,
                background_duck_target_under_voice_db=config.dub_audio.background_duck_target_under_voice_db,
                background_duck_high_coverage_under_voice_db=config.dub_audio.background_duck_high_coverage_under_voice_db,
                background_duck_max_makeup_db=config.dub_audio.background_duck_max_makeup_db,
                background_duck_wideband_when_adaptive=config.dub_audio.background_duck_wideband_when_adaptive,
                source_bed_duck_volume=config.dub_audio.source_bed_duck_volume,
                source_bed_lowpass_hz=config.dub_audio.source_bed_lowpass_hz,
            ),
        ]
    raise ValueError(f"Unknown pipeline preset: {preset}")


def _plan_runners(config: AppConfig, *, render_audio: bool, render_video: bool):
    renderer = _renderer(config, render_audio=render_audio, render_video=render_video)
    dubbing_service = _dubbing_service(config)
    return [
        ExistingArtifactStageRunner(StageName.DOWNLOAD, "source_video", "source_video", "No source_video in task"),
        ExistingArtifactStageRunner(StageName.TRANSCRIBE, "subtitle_rows_json", "subtitle_rows_json", "No subtitle_rows_json in task"),
        ExistingArtifactStageRunner(StageName.TRANSLATE, "translations_json", "translations_json", "No translations_json in task"),
        TtsPrepareStageRunner(audio_config=config.tts_settings().audio_config),
        ExistingArtifactStageRunner(StageName.TTS, "dub_segments_json", "dub_segments_json", "No dub_segments_json in task"),
        AudioMixPlanStageRunner(
            service=dubbing_service,
            renderer=renderer,
            render_audio=render_audio or config.render.render_audio,
            timeline_policy=_timeline_policy(config),
        ),
        ComposePlanStageRunner(
            service=dubbing_service,
            renderer=renderer,
            render_video=render_video or config.render.render_video,
            burn_subtitles=config.render.burn_subtitles,
            video_encoder="h264_nvenc" if config.media.ffmpeg_gpu else "",
            audio_bitrate=config.dub_audio.bitrate,
            audio_sample_rate_hz=config.dub_audio.final_audio_sample_rate,
            audio_channels=config.dub_audio.final_audio_channels,
            video_retime=config.dub_audio.video_retime,
            video_retime_fps_mode=config.dub_audio.video_retime_fps_mode,
            video_speed_warn_min=config.dub_audio.video_speed_warn_min,
            video_speed_warn_max=config.dub_audio.video_speed_warn_max,
            final_loudnorm=config.dub_audio.final_loudnorm,
            final_loudnorm_i=config.dub_audio.final_loudnorm_i,
            final_loudnorm_tp=config.dub_audio.final_loudnorm_tp,
            final_loudnorm_lra=config.dub_audio.final_loudnorm_lra,
            final_smooth=config.dub_audio.final_smooth,
            final_smooth_threshold_db=config.dub_audio.final_smooth_threshold_db,
            final_smooth_ratio=config.dub_audio.final_smooth_ratio,
            final_smooth_attack_ms=config.dub_audio.final_smooth_attack_ms,
            final_smooth_release_ms=config.dub_audio.final_smooth_release_ms,
            final_smooth_makeup_db=config.dub_audio.final_smooth_makeup_db,
            background_ducking=config.dub_audio.background_ducking,
            background_bed_mode=config.dub_audio.background_bed_mode,
            background_duck_volume=config.dub_audio.background_duck_volume,
            background_duck_high_coverage_threshold=config.dub_audio.background_duck_high_coverage_threshold,
            background_duck_high_coverage_volume=config.dub_audio.background_duck_high_coverage_volume,
            background_duck_padding_ms=config.dub_audio.background_duck_padding_ms,
            background_duck_merge_gap_ms=config.dub_audio.background_duck_merge_gap_ms,
            background_duck_transition_ms=config.dub_audio.background_duck_transition_ms,
            background_duck_filter=config.dub_audio.background_duck_filter,
            background_duck_lowpass_hz=config.dub_audio.background_duck_lowpass_hz,
            background_duck_adaptive=config.dub_audio.background_duck_adaptive,
            background_duck_target_under_voice_db=config.dub_audio.background_duck_target_under_voice_db,
            background_duck_high_coverage_under_voice_db=config.dub_audio.background_duck_high_coverage_under_voice_db,
            background_duck_max_makeup_db=config.dub_audio.background_duck_max_makeup_db,
            background_duck_wideband_when_adaptive=config.dub_audio.background_duck_wideband_when_adaptive,
            source_bed_duck_volume=config.dub_audio.source_bed_duck_volume,
            source_bed_lowpass_hz=config.dub_audio.source_bed_lowpass_hz,
        ),
    ]


def _renderer(config: AppConfig, *, render_audio: bool, render_video: bool):
    should_render = render_audio or render_video or config.render.render_audio or config.render.render_video
    if not should_render:
        return None
    return FfmpegDubbingRenderer(ffmpeg_path=config.media.ffmpeg_path)


def _dubbing_service(config: AppConfig) -> DubbingRenderService:
    postprocess_audio = _bool_config(config.tts_audio.get("postprocess_audio"), True)
    return DubbingRenderService(
        sample_rate_hz=config.dub_audio.sample_rate,
        bitrate=config.dub_audio.bitrate,
        clip_lowpass_hz=_int_config(config.tts_audio.get("lowpass_hz"), 6800) if postprocess_audio else 0,
        clip_peak_normalize_dbfs=(
            _float_config(config.tts_audio.get("peak_normalize_dbfs"), -3.0)
            if postprocess_audio
            else None
        ),
        clip_fade_in_ms=config.dub_audio.tts_segment_fade_in_ms,
        clip_fade_out_ms=config.dub_audio.tts_segment_fade_out_ms,
        clip_tail_pad_ms=config.dub_audio.tts_segment_tail_pad_ms,
        clip_tail_pad_counts_in_timeline=config.dub_audio.tts_segment_tail_pad_counts_in_timeline,
        clip_tail_cleanup=config.dub_audio.tts_segment_tail_cleanup,
        clip_tail_cleanup_ms=config.dub_audio.tts_segment_tail_cleanup_ms,
        clip_tail_cleanup_lowpass_hz=config.dub_audio.tts_segment_tail_cleanup_lowpass_hz,
        publish_global_audio_speed=config.dub_audio.publish_global_audio_speed,
        publish_target_video_speed_min=config.dub_audio.publish_target_video_speed_min,
        publish_max_audio_speed=config.dub_audio.publish_max_audio_speed,
        publish_short_video_speed_max=config.dub_audio.publish_short_video_speed_max,
        publish_short_video_speed_hard_max=config.dub_audio.publish_short_video_speed_hard_max,
    )


def _timeline_policy(config: AppConfig) -> TimelinePolicy:
    dub = config.dub_audio
    return TimelinePolicy(
        timeline_mode=dub.publish_timeline_mode,
        lead_in_sec=dub.publish_lead_in_ms / 1000,
        line_gap_sec=dub.publish_line_gap_ms / 1000,
        row_gap_sec=dub.publish_row_gap_ms / 1000,
        tail_pad_sec=dub.publish_tail_pad_ms / 1000,
        min_source_gap_sec=dub.publish_min_source_gap_sec,
        max_source_gap_sec=dub.publish_max_source_gap_sec,
        source_gap_scale=dub.publish_source_gap_scale,
        preserve_source_gaps=True,
        preserve_short_source_windows=dub.publish_preserve_short_source_windows,
        source_window_stretch_max=dub.publish_source_window_stretch_max,
        source_window_borrow_enabled=dub.publish_source_window_borrow_enabled,
        source_window_borrow_max_sec=dub.publish_source_window_borrow_max_sec,
        source_window_borrow_max_ratio=dub.publish_source_window_borrow_max_ratio,
        source_window_borrow_min_seam_sec=dub.publish_source_window_borrow_min_seam_sec,
        source_window_retime_tier2_enabled=dub.publish_source_window_retime_tier2_enabled,
    )


def _media_provider(config: AppConfig) -> FfmpegMediaProvider:
    return FfmpegMediaProvider(ffmpeg_path=config.media.ffmpeg_path, ffprobe_path=config.media.ffprobe_path)


def _asr_provider(config: AppConfig):
    provider = config.asr.provider.lower()
    if provider in {"whisperx", "whisperx-local", "local"}:
        return WhisperRuntimeAsrProvider(name="whisperx")
    if provider in {"whisperx-302", "whisperx_302", "302", "cloud"}:
        return WhisperRuntimeAsrProvider(name="whisperx-302")
    if provider == "elevenlabs":
        return WhisperRuntimeAsrProvider(name="elevenlabs")
    raise ValueError(f"Unsupported whisper.runtime: {config.asr.provider}")


def _vocal_separation_provider(config: AppConfig):
    if not config.demucs.enabled:
        return None
    return DemucsVocalSeparationProvider()


def _llm_client(config: AppConfig) -> OpenAICompatibleLlmClient:
    settings = OpenAICompatibleSettings(
        base_url=config.api.base_url,
        model=config.api.model,
        api_key=config.api.key,
        response_format_json=config.api.llm_support_json,
        timeout_sec=config.api.timeout_sec,
        user_agent=config.api.user_agent,
        trust_env_proxy=config.api.trust_env_proxy,
        proxy_url=config.api.proxy_url,
        stream_responses=config.api.llm_stream,
        max_retries=config.api.max_retries,
        retry_base_delay_sec=config.api.retry_base_delay_sec,
        retry_max_delay_sec=config.api.retry_max_delay_sec,
    )
    transport = RequestsHttpTransport(
        trust_env=settings.trust_env_proxy,
        proxy_url=settings.proxy_url,
    )
    interface = str(config.api.llm_interface or "chat_completions").strip().lower().replace("-", "_")
    if interface in {"codex_responses", "codex_response", "responses", "responses_api"}:
        return CodexResponsesLlmClient(settings, transport)
    if interface not in {"chat_completions", "chat_completion", "chat"}:
        raise ValueError(f"Unsupported LLM interface: {config.api.llm_interface}")
    return OpenAICompatibleLlmClient(settings, transport)


def _tts_provider(config: AppConfig):
    if config.tts_method == "indextts":
        return IndexTtsProvider()
    raise ValueError(f"Unsupported TTS provider: {config.tts_method}")


def _source_url_provider(config: AppConfig):
    provider = config.source.url_provider.lower()
    if provider in {"", "none", "disabled"}:
        return None
    if provider in {"yt-dlp", "ytdlp"}:
        return YtDlpSourceProvider(executable=config.source.yt_dlp_path)
    raise ValueError(f"Unsupported source url provider: {config.source.url_provider}")


def _bool_config(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_config(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_config(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
