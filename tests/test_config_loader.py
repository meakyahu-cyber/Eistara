from __future__ import annotations

from pathlib import Path

from eistara.config import AppConfig, ConfigLoader
from eistara.config.loader import deep_merge, parse_simple_yaml


def test_deep_merge_preserves_nested_defaults() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"x": 9}})

    assert merged == {"a": {"x": 9, "y": 2}}


def test_config_loader_defaults() -> None:
    config = ConfigLoader(local_config=False).load()

    assert config.api.base_url == "https://sub.100xlabs.space/v1"
    assert config.api.model == "claude-opus-4-8"
    assert config.api.timeout_sec == 300
    assert config.api.user_agent == "curl/8.19.0"
    assert config.api.trust_env_proxy is True
    assert config.api.max_retries == 6
    assert config.api.retry_base_delay_sec == 4.0
    assert config.api.retry_max_delay_sec == 60.0
    assert config.target_language == "Simplified Chinese"
    assert config.tts_method == "indextts"
    assert config.indextts["api_url"] == "http://127.0.0.1:8010/tts"
    assert config.youtube.cookies_from_browser == "firefox"
    assert config.source_settings().provider_config["cookies_from_browser"] == "firefox"
    assert config.batch.jobs_dir == Path("jobs")
    assert config.fish_tts["character"] == "AD学姐"
    assert config.fish_tts["character_id_dict"]["丁真"] == "54a5170264694bfc8e9ad98df7bd89c3"


def test_config_loader_default_api_surface_matches_v1_config() -> None:
    data = ConfigLoader(local_config=False).load_dict()

    assert data["api"] == {
        "key": "",
        "base_url": "https://sub.100xlabs.space/v1",
        "model": "claude-opus-4-8",
        "llm_support_json": True,
        "proxy_url": "",
    }


def test_config_loader_reads_private_local_api_key(tmp_path: Path) -> None:
    run_config = tmp_path / "run.yaml"
    run_config.write_text("api:\n  base_url: https://llm.test/v1\n", encoding="utf-8")
    local_config = tmp_path / "config.local.yaml"
    local_config.write_text("api:\n  key: local-secret\n", encoding="utf-8")

    config = ConfigLoader(run_config, local_config=local_config).load()

    assert config.api.base_url == "https://llm.test/v1"
    assert config.api.key == "local-secret"


def test_config_loader_env_api_key_overrides_file(tmp_path: Path, monkeypatch) -> None:
    run_config = tmp_path / "run.yaml"
    run_config.write_text("api:\n  key: file-secret\n", encoding="utf-8")
    monkeypatch.setenv("EISTARA_API_KEY", "env-secret")

    config = ConfigLoader(run_config, local_config=False).load()

    assert config.api.key == "env-secret"


def test_config_loader_get_nested_default() -> None:
    loader = ConfigLoader()

    assert loader.get("translation.publish_fast_chunk_lines") == 30
    assert loader.get("missing.value", "fallback") == "fallback"


def test_config_loader_reads_yaml_and_builds_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
api:
  base_url: https://llm.test/v1
  user_agent: curl/8.19.0
  trust_env_proxy: true
summary_length: 1234
target_language: Simplified Chinese
translation:
  publish_fast_chunk_lines: 7
indextts:
  api_url: http://tts.test/tts
  top_p: 0.6
tts_audio:
  lowpass_hz: 5000
asr:
  provider: whisper
  model: tiny
  language: en
source:
  url_provider: yt-dlp
  output_filename: input.mp4
  yt_dlp_path: yt-dlp-custom
media:
  ffmpeg_path: ffmpeg-custom
render:
  render_audio: true
""".strip(),
        encoding="utf-8",
    )

    config = ConfigLoader(path).load()

    assert config.api.base_url == "https://llm.test/v1"
    assert config.api.user_agent == "curl/8.19.0"
    assert config.api.trust_env_proxy is True
    assert config.translation_settings().max_batch_lines == 7
    assert config.translation_settings().max_batch_chars == 3000
    assert config.translation_settings().summary_length == 1234
    assert config.tts_settings().provider_config["api_url"] == "http://tts.test/tts"
    assert config.tts_settings().provider_config["top_p"] == 0.6
    assert config.tts_settings().audio_config["lowpass_hz"] == 5000
    assert config.asr.provider == "local"
    assert config.asr_settings().model == "tiny"
    assert config.asr_settings().language == "en"
    assert "provider" not in config.asr_settings().provider_config
    assert config.asr_settings().provider_config["cookies_from_browser"] == "firefox"
    assert "subtitle_first" not in config.asr_settings().provider_config
    assert config.asr_settings().provider_config["spacy_model_map"]["en"] == "en_core_web_md"
    assert config.asr_settings().provider_config["language_split_with_space"] == ["en", "es", "fr", "de", "it", "ru"]
    assert config.source.output_filename == "input.mp4"
    assert config.source_settings().output_filename == "input.mp4"
    assert config.source.yt_dlp_path == "yt-dlp-custom"
    assert config.media.ffmpeg_path == "ffmpeg-custom"
    assert config.render.render_audio is True


def test_simple_yaml_parser_reads_scalar_lists() -> None:
    parsed = parse_simple_yaml(
        """
allowed_video_formats:
- mp4
- webm
source:
  yt_dlp_extra_args:
  - --force-ipv4
""".strip()
    )

    assert parsed["allowed_video_formats"] == ["mp4", "webm"]
    assert parsed["source"]["yt_dlp_extra_args"] == ["--force-ipv4"]


def test_config_loader_reads_v1_production_surface(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
display_language: zh-CN
api:
  base_url: https://llm.test/v1
max_workers: 8
batch:
  jobs_dir: batch_jobs
  download_workers: 5
  tts_workers: 2
  dependency_probe: false
  max_auto_requeues: 4
target_language: Simplified Chinese
translation:
  publish_fast_chunk_lines: 9
demucs: true
demucs_segment_minutes: 12
whisper:
  model: large-v3
  language: ja
  runtime: local
  print_progress: true
youtube:
  cookies_from_browser: firefox
  socket_timeout: 33
ytb_resolution: "720"
subtitle:
  display_max_chars_per_line: 18
  display_source_max_chars_per_line: 40
  display_max_lines: 2
burn_subtitles: true
ffmpeg_gpu: true
tts_method: custom_tts
custom_tts:
  python_callable: my_tts.speak
tts_audio:
  merge_micro_lines: false
  lowpass_hz: 5000
indextts:
  prompt_audio_mode: auto_ref
  auto_prompt_target_sec: 15
dub_audio:
  background_bed_mode: source
  final_loudnorm_i: -18
allowed_video_formats:
- mp4
- webm
""".strip(),
        encoding="utf-8",
    )

    config = ConfigLoader(path).load()

    assert config.runtime.max_workers == 8
    assert config.runtime.allowed_video_formats == ("mp4", "webm")
    assert config.batch.stage_worker_limits()["download"] == 5
    assert config.batch.stage_worker_limits()["tts"] == 2
    assert "quality" not in config.batch.stage_worker_limits()
    assert "quality" not in config.batch.stage_idle_timeouts()
    assert config.batch.dependency_probe is False
    assert config.batch.max_auto_requeues == 4
    assert config.asr.model == "large-v3"
    assert config.asr.language == "ja"
    assert config.asr.provider == "local"
    assert config.asr_settings().provider_config["runtime"] == "local"
    assert config.asr_settings().provider_config["print_progress"] is True
    assert config.demucs.enabled is True
    assert config.demucs.segment_minutes == 12
    assert config.demucs.provider == "demucs"
    assert "subtitle_first" not in config.asr_settings().provider_config
    assert config.source.resolution == "720"
    assert config.source_settings().provider_config["cookies_from_browser"] == "firefox"
    assert config.source_settings().provider_config["socket_timeout"] == 33
    assert config.subtitle.display_max_chars_per_line == 18
    assert config.subtitle.display_source_max_chars_per_line == 40
    assert config.subtitle.display_max_lines == 2
    assert config.media.ffmpeg_gpu is True
    assert config.render.burn_subtitles is True
    assert config.tts_settings().method == "custom_tts"
    assert config.tts_settings().provider_config["python_callable"] == "my_tts.speak"
    assert config.tts_settings().audio_config["merge_micro_lines"] is False
    assert config.tts_settings().audio_config["ffmpeg_path"] == config.media.ffmpeg_path
    assert config.indextts["auto_prompt_target_sec"] == 15
    assert config.dub_audio.background_bed_mode == "source"
    assert config.dub_audio.final_loudnorm_i == -18


def test_config_loader_reads_audio_separator_vocal_backend() -> None:
    config = AppConfig.from_dict(
        {
            "demucs": {
                "enabled": True,
                "provider": "audio-separator",
                "segment_minutes": 9,
                "audio_separator_model": "UVR-MDX-NET-Voc_FT.onnx",
                "audio_separator_model_dir": "./models/audio-separator",
            }
        }
    )

    assert config.demucs.enabled is True
    assert config.demucs.provider == "audio-separator"
    assert config.demucs.segment_minutes == 9
    assert config.demucs.audio_separator_model == "UVR-MDX-NET-Voc_FT.onnx"
    assert str(config.demucs.audio_separator_model_dir) == "models\\audio-separator"
    assert config.asr_settings().provider_config["vocal_separation_provider"] == "audio-separator"


def test_config_loader_uses_v1_whisper_runtime_as_asr_route() -> None:
    assert ConfigLoader(local_config=False).load().asr.provider == "local"
    assert ConfigLoader(local_config=False).load_dict().get("asr") is None

    cloud = AppConfig.from_dict({"whisper": {"runtime": "cloud"}})
    elevenlabs = AppConfig.from_dict({"whisper": {"runtime": "elevenlabs"}})

    assert cloud.asr.provider == "cloud"
    assert elevenlabs.asr.provider == "elevenlabs"
