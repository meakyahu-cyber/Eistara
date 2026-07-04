from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any

from eistara.core.jobs import STAGE_ORDER, JobStatus
from eistara.runtime import WEBUI_DEFAULT_PRESET

from apps.webui.backend import WebUiBackend, WebUiSettings
from apps.webui.cover import (
    COVER_TEMPLATES,
    build_contact_sheet,
    build_cover_candidates,
    extract_frames,
    parse_times,
    probe_duration,
    suggest_title_from_video,
)
from apps.webui.task_runner import TaskRunner
from apps.webui.ui_text import LANGUAGE_OPTIONS, normalize_language, ui_t


STAGE_OPTIONS = [stage.value for stage in STAGE_ORDER]
TEXT_STAGES = ("transcribe", "translate")
DUB_STAGES = ("tts_prepare", "tts", "audio_mix", "compose")
TEXT_TARGET_STAGE = "translate"
FULL_TARGET_STAGE = "compose"
SUBTITLE_MODE = "subtitle"
DUBBING_MODE = "dubbing"
MODE_SESSION_KEY = "eistara_delivery_mode"


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit("Streamlit is required for the WebUI. Install streamlit or use the CLI.") from exc

    st.set_page_config(page_title="Eistara", page_icon="docs/logo.svg", layout="wide")
    st.markdown(STYLE, unsafe_allow_html=True)

    state = _sidebar(st)
    backend = WebUiBackend(
        WebUiSettings(
            jobs_dir=Path(state["jobs_dir"]),
            config_path=Path(state["config_path"]) if state["config_path"] else None,
            preset=state["preset"],
        )
    )

    language = state["language"]
    _render_header(st, language)
    mode = str(st.session_state.get(MODE_SESSION_KEY) or "")
    if mode not in {SUBTITLE_MODE, DUBBING_MODE}:
        _mode_selector(st, language)
        return

    _mode_toolbar(st, language, mode)
    if mode == SUBTITLE_MODE:
        _subtitle_mode_placeholder(st, language)
        return

    single_tab, batch_tab, history_tab = st.tabs(
        [ui_t(language, "single_video"), ui_t(language, "batch"), ui_t(language, "history")]
    )
    with single_tab:
        _single_video_tab(st, backend, language, state["preset"])
    with batch_tab:
        _batch_tab(st, backend, language, state["preset"])
    with history_tab:
        _history_tab(st, backend, language)


def _sidebar(st) -> dict[str, str]:
    env_config_text = os.environ.get("EISTARA_CONFIG", "")
    config_text = str(st.session_state.get("runtime_config_path", env_config_text) or "").strip()
    config_path = Path(config_text).expanduser() if config_text else None
    settings_backend = WebUiBackend(WebUiSettings(Path("jobs"), config_path=config_path))
    config = settings_backend.config_dict()
    language = normalize_language(str(_config_get(config, "display_language", "zh-CN")))

    with st.sidebar:
        st.markdown(
            f"""
            <div class="sidebar-brand">
              <div class="sidebar-brand-title">Eistara</div>
              <div class="sidebar-brand-subtitle">{html.escape(ui_t(language, "app_subtitle"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        language_labels = list(LANGUAGE_OPTIONS.keys())
        language_values = list(LANGUAGE_OPTIONS.values())
        current_language_value = language if language in language_values else "zh-CN"
        display_language = st.segmented_control(
            ui_t(language, "language"),
            options=language_labels,
            default=language_labels[language_values.index(current_language_value)],
            key="display_language_switch",
        )
        if display_language and LANGUAGE_OPTIONS[display_language] != current_language_value:
            settings_backend.update_config({"display_language": LANGUAGE_OPTIONS[display_language]})
            st.rerun()

        with st.expander(ui_t(language, "advanced_runtime"), expanded=False):
            st.caption(ui_t(language, "advanced_runtime_hint"))
            preset = WEBUI_DEFAULT_PRESET
            config_override = st.text_input("Config path", value=config_text, key="runtime_config_path")
            if config_override != config_text:
                st.rerun()
        if "preset" not in locals():
            preset = WEBUI_DEFAULT_PRESET
        common_jobs_dir = os.environ.get("EISTARA_JOBS_DIR", str(_config_get(config, "batch.jobs_dir", "jobs")))

        with st.expander(ui_t(language, "llm_configuration"), expanded=False):
            _llm_settings(st, settings_backend, config, language)

        with st.expander(ui_t(language, "subtitles_settings"), expanded=False):
            language_options = _recognition_language_options(language)
            _config_select(
                st,
                settings_backend,
                config,
                "whisper.language",
                ui_t(language, "recognition_language"),
                list(language_options),
                format_func=lambda value: language_options.get(value, value),
            )
            runtime_labels = {
                "local": ui_t(language, "whisper_runtime_local"),
                "cloud": ui_t(language, "whisper_runtime_cloud"),
                "elevenlabs": ui_t(language, "whisper_runtime_elevenlabs"),
            }
            current_runtime = str(_config_get(config, "whisper.runtime", "local") or "local")
            _config_select(
                st,
                settings_backend,
                config,
                "whisper.runtime",
                ui_t(language, "whisper_runtime"),
                ["local", "cloud", "elevenlabs"],
                format_func=lambda value: runtime_labels.get(value, value),
            )
            if current_runtime == "local":
                _config_text(st, settings_backend, config, "whisper.model", ui_t(language, "whisper_model"))
            elif current_runtime == "cloud":
                _config_text(
                    st,
                    settings_backend,
                    config,
                    "whisper.whisperX_302_api_key",
                    ui_t(language, "whisper_302_key"),
                    password=True,
                )
            elif current_runtime == "elevenlabs":
                _config_text(
                    st,
                    settings_backend,
                    config,
                    "whisper.elevenlabs_api_key",
                    ui_t(language, "elevenlabs_asr_key"),
                    password=True,
                )
            _render_asr_runtime_status(st, settings_backend, language)
            target_language = str(_config_get(config, "target_language", "Simplified Chinese") or "")
            target_display = ui_t(language, "target_simplified_chinese") if language == "zh-CN" and target_language == "Simplified Chinese" else target_language
            value = st.text_input(ui_t(language, "target_language"), value=target_display, key="config_target_language")
            if value != target_display:
                settings_backend.update_config({"target_language": value})
                st.rerun()
            _vocal_separation_settings(st, settings_backend, config, language)

        with st.expander(ui_t(language, "dubbing_settings"), expanded=False):
            _dubbing_settings(st, settings_backend, config, language)

        _youtube_cookie_settings(st, settings_backend, language)
        _runtime_status(st, settings_backend, language)

    return {
        "config_path": str(config_path) if config_path else "",
        "preset": preset,
        "jobs_dir": common_jobs_dir,
        "language": language,
    }


def _mode_selector(st, language: str) -> None:
    _section_title(st, ui_t(language, "choose_delivery_mode"), ui_t(language, "delivery_mode_caption"))
    st.markdown(
        f"""
        <div class="eistara-mode-note">
          {html.escape(ui_t(language, "mode_strategy_notice"))}
        </div>
        """,
        unsafe_allow_html=True,
    )
    subtitle_col, dubbing_col = st.columns(2, gap="large")
    with subtitle_col:
        st.markdown(
            f"""
            <div class="eistara-mode-card subtitle">
              <div class="eistara-mode-kicker">{html.escape(ui_t(language, "subtitle_mode_kicker"))}</div>
              <div class="eistara-mode-title">{html.escape(ui_t(language, "subtitle_mode_title"))}</div>
              <div class="eistara-mode-body">{html.escape(ui_t(language, "subtitle_mode_body"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(ui_t(language, "enter_subtitle_mode"), key="enter_subtitle_mode", use_container_width=True):
            st.session_state[MODE_SESSION_KEY] = SUBTITLE_MODE
            st.rerun()
    with dubbing_col:
        st.markdown(
            f"""
            <div class="eistara-mode-card dubbing">
              <div class="eistara-mode-kicker">{html.escape(ui_t(language, "dubbing_mode_kicker"))}</div>
              <div class="eistara-mode-title">{html.escape(ui_t(language, "dubbing_mode_title"))}</div>
              <div class="eistara-mode-body">{html.escape(ui_t(language, "dubbing_mode_body"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(ui_t(language, "enter_dubbing_mode"), key="enter_dubbing_mode", type="primary", use_container_width=True):
            st.session_state[MODE_SESSION_KEY] = DUBBING_MODE
            st.rerun()


def _mode_toolbar(st, language: str, mode: str) -> None:
    mode_label = ui_t(language, "subtitle_mode_title") if mode == SUBTITLE_MODE else ui_t(language, "dubbing_mode_title")
    label_col, action_col = st.columns([4, 1], gap="large")
    with label_col:
        st.markdown(
            f"""
            <div class="eistara-mode-current">
              <span>{html.escape(ui_t(language, "current_delivery_mode"))}</span>
              <strong>{html.escape(mode_label)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        if st.button(ui_t(language, "switch_delivery_mode"), key="switch_delivery_mode", use_container_width=True):
            st.session_state.pop(MODE_SESSION_KEY, None)
            st.rerun()


def _subtitle_mode_placeholder(st, language: str) -> None:
    _section_title(st, ui_t(language, "subtitle_workspace"), ui_t(language, "subtitle_workspace_caption"))
    with st.container(border=True):
        st.info(ui_t(language, "subtitle_mode_notice"))
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.markdown(f"**{html.escape(ui_t(language, 'subtitle_delivery_focus'))}**")
            st.markdown(_steps_html([
                ui_t(language, "subtitle_focus_line_breaks"),
                ui_t(language, "subtitle_focus_char_limits"),
                ui_t(language, "subtitle_focus_reading_speed"),
            ]), unsafe_allow_html=True)
        with right:
            st.markdown(f"**{html.escape(ui_t(language, 'subtitle_mode_boundary'))}**")
            st.warning(ui_t(language, "subtitle_no_dubbing_warning"))


def _llm_settings(st, backend: WebUiBackend, config: dict[str, Any], language: str) -> None:
    active = backend.active_config()
    api = active.get("api", {})
    st.caption(
        "Active config: "
        f"{active.get('path')} | "
        f"base_url={api.get('base_url') or '-'} | "
        f"model={api.get('model') or '-'} | "
        f"key={'set' if api.get('has_key') else 'missing'} | "
        f"proxy={api.get('proxy_url') or '-'}"
    )
    _config_text(st, backend, config, "api.key", ui_t(language, "api_key"), password=True)
    _config_text(st, backend, config, "api.base_url", ui_t(language, "base_url"))
    _config_text(st, backend, config, "api.proxy_url", "LLM proxy URL")

    if st.button(ui_t(language, "fetch_model_list"), key="fetch_models", use_container_width=True):
        try:
            with st.spinner(ui_t(language, "fetching_models")):
                result = backend.list_llm_models()
            models = list(result.get("models") or [])
            st.session_state["_model_list"] = models
            if models:
                st.success(ui_t(language, "fetched_models").replace("{n}", str(len(models))))
            else:
                st.warning(ui_t(language, "fetch_models_failed"))
        except Exception as exc:
            st.session_state["_model_list"] = []
            st.error(f"{ui_t(language, 'fetch_models_failed')} {exc}")

    current_model = str(_config_get(config, "api.model", "") or "")
    model_list = list(st.session_state.get("_model_list") or [])
    model_updated = _model_selector(st, backend, current_model, model_list, language)
    if model_updated:
        st.rerun()

    if st.button(ui_t(language, "check_api"), key="api_check", use_container_width=True):
        try:
            with st.spinner(ui_t(language, "checking_api")):
                result = backend.check_llm_api()
            if result.get("ok"):
                st.success(ui_t(language, "api_key_valid"))
            else:
                st.error(ui_t(language, "api_key_invalid"))
                st.code(json.dumps(result.get("response"), ensure_ascii=False, indent=2), language="json")
        except Exception as exc:
            st.error(f"{ui_t(language, 'api_key_invalid')}: {exc}")

    _config_toggle(st, backend, config, "api.llm_support_json", ui_t(language, "llm_json_support"))


def _model_selector(st, backend: WebUiBackend, current_model: str, model_list: list[str], language: str) -> bool:
    options = _model_options(current_model, model_list)
    if model_list:
        index = options.index(current_model) if current_model in options else None
        selected_model = st.selectbox(
            ui_t(language, "model"),
            options=options,
            index=index,
            placeholder=ui_t(language, "search_or_enter_model"),
            key="config_api_model_select",
            accept_new_options=True,
            filter_mode="fuzzy",
        )
        if selected_model and selected_model != current_model:
            backend.update_config({"api.model": selected_model})
            return True
        return False

    value = st.text_input(
        ui_t(language, "model"),
        value=current_model,
        placeholder=ui_t(language, "search_or_enter_model"),
        key="config_api_model",
    )
    if value != current_model:
        backend.update_config({"api.model": value})
        return True
    return False


def _model_options(current_model: str, model_list: list[str]) -> list[str]:
    options = []
    if current_model:
        options.append(current_model)
    options.extend(model for model in model_list if model and model not in options)
    return options


def _vocal_separation_settings(st, backend: WebUiBackend, config: dict[str, Any], language: str) -> None:
    current = _vocal_separation_config(config)
    enabled = st.toggle(ui_t(language, "vocal_separation"), value=bool(current["enabled"]), key="config_demucs_enabled")
    if enabled != current["enabled"]:
        _update_vocal_separation_config(backend, current | {"enabled": enabled})
        st.rerun()


def _vocal_separation_config(config: dict[str, Any]) -> dict[str, Any]:
    demucs = config.get("demucs")
    if isinstance(demucs, dict):
        return {
            "enabled": bool(demucs.get("enabled", True)),
            "segment_minutes": float(demucs.get("segment_minutes") or config.get("demucs_segment_minutes") or 30),
        }
    return {
        "enabled": bool(demucs),
        "segment_minutes": float(config.get("demucs_segment_minutes") or 30),
    }


def _update_vocal_separation_config(backend: WebUiBackend, value: dict[str, Any]) -> None:
    backend.update_config(
        {
            "demucs": {
                "enabled": bool(value.get("enabled")),
                "segment_minutes": float(value.get("segment_minutes") or 30),
            }
        }
    )


def _dubbing_settings(st, backend: WebUiBackend, config: dict[str, Any], language: str) -> None:
    local_tts_methods = ["indextts", "custom_tts", "gpt_sovits"]
    network_tts_methods = ["edge_tts", "openai_tts", "azure_tts", "fish_tts", "sf_fish_tts", "sf_cosyvoice2", "f5tts"]
    current_tts_method = str(_config_get(config, "tts_method", "indextts"))
    current_tts_source = "local" if current_tts_method in local_tts_methods else "remote"
    source_options = {
        ui_t(language, "tts_source_local"): "local",
        ui_t(language, "tts_source_remote"): "remote",
    }
    selected_tts_source_label = st.selectbox(
        ui_t(language, "tts_source"),
        options=list(source_options),
        index=list(source_options.values()).index(current_tts_source),
        key="config_tts_source",
    )
    selected_tts_source = source_options[selected_tts_source_label]
    tts_methods = local_tts_methods if selected_tts_source == "local" else network_tts_methods
    if current_tts_method not in tts_methods:
        backend.update_config({"tts_method": tts_methods[0]})
        st.rerun()
    selected_tts = st.selectbox(
        ui_t(language, "tts_method"),
        options=tts_methods,
        index=tts_methods.index(current_tts_method),
        key="config_tts_method",
    )
    if selected_tts != current_tts_method:
        backend.update_config({"tts_method": selected_tts})
        st.rerun()

    with st.expander(ui_t(language, "tts_audio_rules"), expanded=False):
        _config_toggle(st, backend, config, "tts_audio.merge_micro_lines", ui_t(language, "tts_micro_merge"))
        _config_number(st, backend, config, "tts_audio.merge_micro_line_chars", ui_t(language, "tts_micro_chars"), 1, 20)
        _config_toggle(st, backend, config, "tts_audio.postprocess_audio", ui_t(language, "tts_audio_postprocess"))
        _config_toggle(st, backend, config, "tts_audio.trim_silence", ui_t(language, "tts_audio_trim_silence"))
        _config_number(st, backend, config, "tts_audio.lowpass_hz", ui_t(language, "tts_audio_lowpass"), 0, 12000, step=100)
        _config_float_slider(st, backend, config, "tts_audio.peak_normalize_dbfs", ui_t(language, "tts_audio_peak"), -12.0, 0.0, step=0.5)

    st.caption(f"{ui_t(language, 'dubbing_timeline')}: {ui_t(language, 'timeline_publish')}")
    with st.expander(ui_t(language, "publish_mix"), expanded=False):
        _config_float_slider(st, backend, config, "dub_audio.publish_target_video_speed_min", ui_t(language, "minimum_video_speed"), 0.75, 1.0, step=0.01)
        _config_float_slider(st, backend, config, "dub_audio.publish_max_audio_speed", ui_t(language, "max_global_dub_speed"), 1.0, 1.35, step=0.01)
        bed_modes = {
            "separated": ui_t(language, "background_separated"),
            "source_ducked": ui_t(language, "background_source_ducked"),
        }
        _config_select(
            st,
            backend,
            config,
            "dub_audio.background_bed_mode",
            ui_t(language, "background_bed"),
            list(bed_modes),
            format_func=lambda value: bed_modes.get(value, value),
        )
        if _config_get(config, "dub_audio.background_bed_mode", "separated") == "source_ducked":
            st.warning(ui_t(language, "source_bed_warning"))
            _config_float_slider(st, backend, config, "dub_audio.source_bed_duck_volume", ui_t(language, "source_duck_volume"), 0.04, 0.20, step=0.01)
            _config_number(st, backend, config, "dub_audio.source_bed_lowpass_hz", ui_t(language, "source_duck_lowpass"), 2400, 5000, step=100)
        else:
            _config_float_slider(st, backend, config, "dub_audio.background_duck_high_coverage_volume", ui_t(language, "high_coverage_background"), 0.30, 1.0, step=0.05)
        _config_toggle(st, backend, config, "dub_audio.final_loudnorm", ui_t(language, "final_loudness_normalize"))
        _config_float_slider(st, backend, config, "dub_audio.final_loudnorm_i", ui_t(language, "target_lufs"), -20.0, -14.0, step=0.5)

    if selected_tts == "indextts":
        _config_text(st, backend, config, "indextts.api_url", "IndexTTS API URL")
        _config_text(st, backend, config, "indextts.prompt_audio", "IndexTTS prompt_audio")
    elif selected_tts == "custom_tts":
        _config_select(st, backend, config, "custom_tts.mode", "custom_tts.mode", ["python_callable", "command", "placeholder"])
        _config_text(st, backend, config, "custom_tts.python_callable", "custom_tts.python_callable")
        _config_text(st, backend, config, "custom_tts.placeholder_audio", "custom_tts.placeholder_audio")
    elif selected_tts == "sf_fish_tts":
        _config_text(st, backend, config, "sf_fish_tts.api_key", "SiliconFlow API Key", password=True)
        _config_text(st, backend, config, "sf_fish_tts.voice", "Voice")
    elif selected_tts == "openai_tts":
        _config_text(st, backend, config, "openai_tts.api_key", "302ai API", password=True)
        _config_text(st, backend, config, "openai_tts.voice", "OpenAI Voice")
    elif selected_tts == "fish_tts":
        _config_text(st, backend, config, "fish_tts.api_key", "302ai API", password=True)
    elif selected_tts == "azure_tts":
        _config_text(st, backend, config, "azure_tts.api_key", "302ai API", password=True)
        _config_text(st, backend, config, "azure_tts.voice", "Azure Voice")
    elif selected_tts == "gpt_sovits":
        _config_text(st, backend, config, "gpt_sovits.character", "SoVITS Character")
        _config_number(st, backend, config, "gpt_sovits.refer_mode", "Refer Mode", 1, 3)
    elif selected_tts == "edge_tts":
        _config_text(st, backend, config, "edge_tts.voice", "Edge TTS Voice")
    elif selected_tts == "sf_cosyvoice2":
        _config_text(st, backend, config, "sf_cosyvoice2.api_key", "SiliconFlow API Key", password=True)
    elif selected_tts == "f5tts":
        _config_text(st, backend, config, "f5tts.302_api", "302ai API", password=True)


def _youtube_cookie_settings(st, backend: WebUiBackend, language: str) -> None:
    with st.expander(ui_t(language, "youtube_cookies"), expanded=False):
        try:
            youtube_cookies = backend.youtube_cookies()
        except Exception as exc:
            st.warning(str(exc))
            return
        current = youtube_cookies["configured_browser"] or ui_t(language, "not_configured")
        if youtube_cookies["configured_profile"]:
            current = f"{current}:{youtube_cookies['configured_profile']}"
        st.caption(f"{ui_t(language, 'current_cookie_source')}: {current}")
        candidates = youtube_cookies["candidates"]
        browsers = []
        for candidate in candidates:
            browser = str(candidate.get("browser") or "")
            if browser and browser not in browsers:
                browsers.append(browser)
        options = ["auto"] + browsers
        selected_browser = st.selectbox(ui_t(language, "cookie_source"), options, index=0, key="youtube_cookie_browser")
        profile = st.text_input(ui_t(language, "browser_profile"), value="", key="youtube_cookie_profile")
        if st.button(ui_t(language, "apply_cookie_source"), use_container_width=True, key="youtube_cookie_apply"):
            try:
                backend.configure_youtube_cookies(browser=selected_browser, profile=profile)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _runtime_status(st, backend: WebUiBackend, language: str) -> None:
    try:
        health = backend.health()
    except Exception:
        return
    tts = health.get("tts") or {}
    ok = tts.get("ok")
    tts_state = "unknown" if ok is None else "online" if ok else "offline"
    st.caption(f"{ui_t(language, 'indextts_status')}: {tts_state}")
    _render_dependency_status(st, health.get("model_dependencies") or {})


def _render_asr_runtime_status(st, backend: WebUiBackend, language: str) -> None:
    try:
        health = backend.health()
    except Exception as exc:
        st.caption(f"{ui_t(language, 'asr_status')}: {exc}")
        return
    report = health.get("model_dependencies") or {}
    rows = [
        item
        for item in report.get("items") or []
        if item.get("component") in {"asr", "vocal_separation"}
    ]
    asr_rows = [item for item in rows if item.get("component") == "asr"]
    required_missing = [item for item in asr_rows if item.get("required") and item.get("ok") is False]
    if required_missing:
        st.caption(f"{ui_t(language, 'asr_status')}: {ui_t(language, 'asr_status_missing')}")
    elif asr_rows:
        st.caption(f"{ui_t(language, 'asr_status')}: {ui_t(language, 'asr_status_ready')}")
    else:
        st.caption(f"{ui_t(language, 'asr_status')}: {ui_t(language, 'asr_status_unknown')}")
    with st.expander(ui_t(language, "asr_dependency_check"), expanded=False):
        if rows:
            st.dataframe(
                [
                    {
                        "component": item.get("component"),
                        "dependency": item.get("dependency"),
                        "mode": item.get("mode"),
                        "required": item.get("required"),
                        "status": item.get("status"),
                        "path": item.get("path"),
                    }
                    for item in rows
                ],
                hide_index=True,
                height=min(260, 38 + 35 * len(rows)),
                width="stretch",
            )
        else:
            st.caption(ui_t(language, "asr_status_unknown"))


def _render_dependency_status(st, report: dict[str, Any]) -> None:
    counts = report.get("counts") or {}
    missing_required = int(counts.get("missing_required") or 0)
    optional_missing = int(counts.get("optional_missing") or 0)
    unknown = int(counts.get("unknown") or 0)
    state = "ok" if missing_required == 0 else f"{missing_required} required missing"
    suffix = []
    if optional_missing:
        suffix.append(f"{optional_missing} optional missing")
    if unknown:
        suffix.append(f"{unknown} unknown")
    st.caption("Dependencies: " + state + (f" ({', '.join(suffix)})" if suffix else ""))
    with st.expander("Dependency check", expanded=False):
        rows = []
        for item in report.get("items") or []:
            rows.append(
                {
                    "component": item.get("component"),
                    "dependency": item.get("dependency"),
                    "mode": item.get("mode"),
                    "required": item.get("required"),
                    "status": item.get("status"),
                    "path": item.get("path"),
                }
            )
        if rows:
            st.dataframe(rows, hide_index=True, height=min(420, 38 + 35 * len(rows)), width="stretch")
        else:
            st.caption("No dependency report available.")


def _single_video_tab(st, backend: WebUiBackend, language: str, preset: str) -> None:
    config = backend.config_dict()
    job_id = backend.latest_active_job_id()
    detail = backend.job_detail(job_id) if job_id else None

    if detail:
        _workflow_status_strip(st, detail, language)

    source_col, flow_col = st.columns([1.05, 1], gap="large")
    with source_col:
        _source_section(st, backend, config, detail, language, preset)
        if detail and _stage_done(detail, "download"):
            with st.expander(ui_t(language, "cover"), expanded=False):
                _cover_section(st, detail, language)
    with flow_col:
        _subtitle_section(st, backend, detail, language, preset)
        _dubbing_section(st, backend, detail, language, preset)


def _source_section(st, backend: WebUiBackend, config: dict[str, Any], detail: dict[str, Any] | None, language: str, preset: str) -> None:
    _section_title(st, ui_t(language, "source"), ui_t(language, "download_or_upload"))
    with st.container(border=True):
        if detail and _stage_done(detail, "download"):
            st.success(ui_t(language, "source_ready"))
            st.caption(detail["job_id"])
            _render_outputs(st, detail["outputs"], language, kinds={"video"}, preview_role="source_video", title=None)
            if st.button(ui_t(language, "delete_source"), key="single_delete_source", use_container_width=True):
                backend.delete_active_job(str(detail["job_id"]))
                st.rerun()
            return

        if detail:
            st.info(ui_t(language, "source_task_ready"))
            _render_job_progress(st, detail, language)

        col1, col2 = st.columns([4, 1])
        with col1:
            source_url = st.text_input(ui_t(language, "download_url"), placeholder="https://www.youtube.com/watch?v=...", key="single_source_url")
        with col2:
            resolution = st.selectbox(ui_t(language, "resolution"), ["1080", "360", "best"], index=0, key="single_resolution")

        if st.button(ui_t(language, "download_video"), key="single_download_video", type="primary", use_container_width=True):
            if not source_url.strip():
                st.error(ui_t(language, "add_video_first"))
            else:
                try:
                    with st.spinner(ui_t(language, "download_video")):
                        backend.create_single_job(
                            source_url.strip(),
                            resolution=resolution,
                            source_language=str(_config_get(config, "whisper.language", "en")),
                            target_language=str(_config_get(config, "target_language", "Simplified Chinese")),
                        )
                        _run_stage_checked(backend, "download", preset)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

        upload = st.file_uploader(
            ui_t(language, "upload_video"),
            type=list(_config_get(config, "allowed_video_formats", ["mp4", "mov", "avi", "mkv", "flv", "wmv", "webm"]))
            + list(_config_get(config, "allowed_audio_formats", ["wav", "mp3", "flac", "m4a"])),
            key="single_upload",
        )
        if upload is not None and st.button(ui_t(language, "use_uploaded_file"), key="single_use_upload", use_container_width=True):
            try:
                with st.spinner(ui_t(language, "use_uploaded_file")):
                    path = backend.save_upload_source(upload.name, upload.getvalue())
                    backend.create_single_job(
                        str(path),
                        resolution=resolution,
                        source_language=str(_config_get(config, "whisper.language", "en")),
                        target_language=str(_config_get(config, "target_language", "Simplified Chinese")),
                    )
                    _run_stage_checked(backend, "download", preset)
                st.rerun()
            except Exception as exc:
                st.error(str(exc))


def _workflow_status_strip(st, detail: dict[str, Any], language: str) -> None:
    state = detail.get("state") or {}
    completed = set(state.get("completed_stages") or [])
    current = str(state.get("current_stage") or state.get("failed_stage") or "")
    status = str(state.get("status") or "")
    items = []
    for stage in STAGE_ORDER:
        stage_value = stage.value
        state_class = "done" if stage_value in completed else "current" if stage_value == current else "pending"
        if status == JobStatus.FAILED.value and stage_value == current:
            state_class = "failed"
        items.append(
            f"""
            <div class="eistara-flow-step eistara-flow-{state_class}">
              <span>{html.escape(_display_stage(stage_value, language))}</span>
            </div>
            """
        )
    st.markdown(
        f"""
        <div class="eistara-flow-strip">
          <div class="eistara-flow-job">{html.escape(str(detail.get("job_id") or ""))}</div>
          <div class="eistara-flow-steps">{"".join(items)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _cover_section(st, detail: dict[str, Any] | None, language: str) -> None:
    _section_title(st, ui_t(language, "cover"), ui_t(language, "cover_caption"))
    source_video = _source_video_path(detail)
    if source_video is None:
        st.info(ui_t(language, "cover_source_missing"))
        return

    output_dir = Path(str(detail.get("output_dir") or Path(str(detail["job_dir"])) / "output")) / "covers"
    contact_sheet, covers = _existing_cover_paths(output_dir)
    default_title = suggest_title_from_video(source_video)
    control_col, preview_col = st.columns([0.95, 1.25], gap="large")
    with control_col:
        title = st.text_input(
            ui_t(language, "cover_title"),
            value=default_title,
            placeholder=ui_t(language, "cover_title_placeholder"),
            key=f"cover_title_{detail['job_id']}",
        )
        template_labels = {
            "cinema": ui_t(language, "cover_template_cinema"),
            "news": ui_t(language, "cover_template_news"),
            "clean": ui_t(language, "cover_template_clean"),
        }
        template = st.selectbox(
            ui_t(language, "cover_template"),
            options=list(COVER_TEMPLATES),
            format_func=lambda value: template_labels.get(value, value),
            key=f"cover_template_{detail['job_id']}",
        )
        subtitle = st.text_input(
            ui_t(language, "cover_subtitle"),
            value="",
            placeholder=ui_t(language, "cover_subtitle_placeholder"),
            key=f"cover_subtitle_{detail['job_id']}",
        )
        accent = st.text_input(
            ui_t(language, "cover_accent"),
            value="",
            placeholder=ui_t(language, "cover_accent_placeholder"),
            key=f"cover_accent_{detail['job_id']}",
        )
        if st.button(ui_t(language, "generate_covers"), key=f"generate_cover_candidates_{detail['job_id']}", use_container_width=True):
            try:
                with st.spinner(ui_t(language, "generate_covers")):
                    _generate_covers(
                        source_video,
                        output_dir,
                        title.strip() or default_title,
                        subtitle.strip(),
                        accent.strip(),
                        str(template),
                    )
                st.success(ui_t(language, "cover_ready"))
                st.rerun()
            except Exception as exc:
                st.error(f"{ui_t(language, 'cover_failed')}: {exc}")

    with preview_col:
        if contact_sheet:
            st.markdown(f"**{ui_t(language, 'cover_contact_sheet')}**")
            st.image(str(contact_sheet), use_container_width=True)
        if covers:
            st.markdown(f"**{ui_t(language, 'cover_candidates')}**")
            for index in range(0, len(covers), 2):
                columns = st.columns(2)
                for column, cover_path in zip(columns, covers[index : index + 2]):
                    with column:
                        st.image(str(cover_path), use_container_width=True)
                        st.download_button(
                            ui_t(language, "cover_download"),
                            data=cover_path.read_bytes(),
                            file_name=cover_path.name,
                            mime="image/jpeg",
                            key=f"download_{detail['job_id']}_{cover_path.stem}",
                            use_container_width=True,
                        )


def _generate_covers(video_path: Path, output_dir: Path, title: str, subtitle: str, accent: str, template: str) -> None:
    duration = probe_duration(video_path)
    frame_dir = output_dir / "frames"
    frames = extract_frames(video_path, frame_dir, parse_times(None, duration))
    build_contact_sheet(frames, output_dir / "cover_contact.jpg")
    build_cover_candidates(frames, output_dir=output_dir, title=title, subtitle=subtitle, accent=accent, template=template)


def _existing_cover_paths(output_dir: Path) -> tuple[Path | None, list[Path]]:
    contact_sheet = output_dir / "cover_contact.jpg"
    covers = sorted(path for path in output_dir.glob("cover_*.jpg") if path.name != "cover_contact.jpg")
    return (
        contact_sheet if contact_sheet.exists() else None,
        [path for path in covers if path.exists()],
    )


def _source_video_path(detail: dict[str, Any] | None) -> Path | None:
    if not detail or not _stage_done(detail, "download"):
        return None
    for item in detail.get("outputs") or []:
        if item.get("role") == "source_video" and item.get("exists"):
            return Path(str(item["path"]))
    return None


def _subtitle_section(st, backend: WebUiBackend, detail: dict[str, Any] | None, language: str, preset: str) -> None:
    _section_title(st, ui_t(language, "translate"))
    runner = TaskRunner.get(st.session_state, "_single_text_runner")
    with st.container(border=True):
        st.markdown(
            _steps_html(["WhisperX word-level transcription", "Publish-first translation and subtitles"]),
            unsafe_allow_html=True,
        )
        if not detail or not _stage_done(detail, "download"):
            st.info(ui_t(language, "create_source_first"))
            return
        if _stage_done(detail, TEXT_TARGET_STAGE):
            st.success(ui_t(language, "subtitles_done"))
            _render_outputs(st, detail["outputs"], language, kinds={"subtitle", "data"}, title=ui_t(language, "subtitle_outputs"))
            return
        if runner.is_active or runner.is_done:
            _task_control_panel(st, "_single_text_runner", language)
        elif st.button(ui_t(language, "start_subtitles"), key="single_start_subtitles", type="primary", use_container_width=True):
            runner.start(_stage_steps(backend, TEXT_STAGES, preset))
            st.rerun()


def _dubbing_section(st, backend: WebUiBackend, detail: dict[str, Any] | None, language: str, preset: str) -> None:
    _section_title(st, ui_t(language, "dub"))
    runner = TaskRunner.get(st.session_state, "_single_dub_runner")
    with st.container(border=True):
        st.markdown(
            _steps_html(["Generate audio tasks and chunks", "Extract reference audio", "Generate and merge audio files", "Merge final audio into video"]),
            unsafe_allow_html=True,
        )
        if not detail or not _stage_done(detail, "download"):
            st.info(ui_t(language, "create_source_first"))
            return
        if not _stage_done(detail, TEXT_TARGET_STAGE):
            st.info(ui_t(language, "finish_subtitles_first"))
            return
        if _stage_done(detail, FULL_TARGET_STAGE):
            st.success(ui_t(language, "dubbing_done"))
            _render_outputs(st, detail["outputs"], language, kinds={"video", "audio", "subtitle"}, preview_role="dub_video", title=ui_t(language, "final_outputs"))
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(ui_t(language, "rebuild_mix"), key="single_rebuild_mix", use_container_width=True):
                    _reset_and_start_runner(st, backend, detail["job_id"], "audio_mix", DUB_STAGES[-2:], preset, "_single_dub_runner")
            with col2:
                if st.button(ui_t(language, "regenerate_tts"), key="single_regenerate_tts", use_container_width=True):
                    _reset_and_start_runner(st, backend, detail["job_id"], "tts", DUB_STAGES[1:], preset, "_single_dub_runner")
            with col3:
                if detail.get("state", {}).get("status") == JobStatus.DONE.value:
                    if st.button(ui_t(language, "archive_to_history"), key="single_archive_to_history", use_container_width=True):
                        backend.archive_job(detail["job_id"])
                        st.success(ui_t(language, "archived_to_history"))
                        st.rerun()
            return
        if runner.is_active or runner.is_done:
            _task_control_panel(st, "_single_dub_runner", language)
        elif st.button(ui_t(language, "start_dubbing"), key="single_start_dubbing", type="primary", use_container_width=True):
            runner.start(_stage_steps(backend, DUB_STAGES, preset))
            st.rerun()


def _reset_and_start_runner(st, backend: WebUiBackend, job_id: str, reset_stage: str, stages: tuple[str, ...], preset: str, runner_key: str) -> None:
    backend.reset_from_stage(job_id, reset_stage, preset=preset)
    runner = TaskRunner.get(st.session_state, runner_key)
    runner.start(_stage_steps(backend, stages, preset))
    st.rerun()


def _stage_steps(backend: WebUiBackend, stages: tuple[str, ...], preset: str) -> list[tuple[str, Any]]:
    return [(stage, lambda target=stage: _run_stage_checked(backend, target, preset)) for stage in stages]


def _run_stage_checked(backend: WebUiBackend, stage: str, preset: str) -> dict[str, Any]:
    result = backend.run_until_stage(stage, preset=preset)
    if result.get("status") != "reached":
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def _task_control_panel(st, runner_key: str, language: str) -> None:
    runner = TaskRunner.get(st.session_state, runner_key)
    if runner.state == "idle":
        return
    step_text = f"({runner.current_step + 1}/{runner.total_steps}) {runner.current_label}" if runner.current_step >= 0 else ""
    if runner.is_active:
        if runner.state == "paused":
            st.warning(f"{ui_t(language, 'paused')} {step_text}")
        else:
            st.info(f"{ui_t(language, 'running')} {step_text}")
        st.progress(runner.progress)
        col1, col2 = st.columns(2)
        with col1:
            if runner.state == "paused":
                if st.button(ui_t(language, "resume"), key=f"{runner_key}_resume", use_container_width=True):
                    runner.resume()
                    st.rerun()
            else:
                if st.button(ui_t(language, "pause"), key=f"{runner_key}_pause", use_container_width=True):
                    runner.pause()
                    st.rerun()
        with col2:
            if st.button(ui_t(language, "stop"), key=f"{runner_key}_stop", use_container_width=True, type="primary"):
                runner.stop()
                st.rerun()
    elif runner.state == "completed":
        st.success(ui_t(language, "task_done"))
        st.progress(1.0)
        runner.reset()
        time.sleep(0.3)
        st.rerun()
    elif runner.state == "stopped":
        st.warning(f"{ui_t(language, 'task_stopped')} {step_text}")
        if st.button(ui_t(language, "ok"), key=f"{runner_key}_ack_stop", use_container_width=True):
            runner.reset()
            st.rerun()
    elif runner.state == "error":
        st.error(f"{ui_t(language, 'task_error')}: {runner.error_msg}")
        if st.button(ui_t(language, "ok"), key=f"{runner_key}_ack_error", use_container_width=True):
            runner.reset()
            st.rerun()


def _batch_tab(st, backend: WebUiBackend, language: str, preset: str) -> None:
    config = backend.config_dict()
    _section_title(st, ui_t(language, "batch"), ui_t(language, "app_subtitle"))
    with st.container(border=True):
        st.caption(ui_t(language, "batch_concurrency_hint"))
        st.markdown(
            f'<div class="eistara-inline-summary">{html.escape(_batch_config_summary(config, language))}</div>',
            unsafe_allow_html=True,
        )
        jobs_dir_text = st.text_input(
            ui_t(language, "batch_jobs_dir"),
            value=str(backend.jobs_dir),
            key="batch_jobs_dir_display",
            disabled=True,
        )
        _ = jobs_dir_text
        sources_text = st.text_area(
            ui_t(language, "batch_sources"),
            height=140,
            placeholder=ui_t(language, "batch_sources_placeholder"),
            key="batch_sources",
        )
        cols = st.columns(3)
        with cols[0]:
            resolution = st.text_input(ui_t(language, "resolution"), value=str(_config_get(config, "ytb_resolution", _config_get(config, "source.resolution", "1080"))), key="batch_resolution")
        with cols[1]:
            language_options = _source_language_options(language)
            source_values = list(language_options.values())
            default_source_language = str(_config_get(config, "whisper.language", "en"))
            if default_source_language not in source_values:
                default_source_language = "en"
            selected_source_language = st.selectbox(
                ui_t(language, "source_language"),
                options=list(language_options),
                index=source_values.index(default_source_language),
                key="batch_source_language",
            )
            source_language = language_options[selected_source_language]
        with cols[2]:
            default_target_language = str(_config_get(config, "target_language", "Simplified Chinese"))
            if language == "zh-CN" and default_target_language == "Simplified Chinese":
                default_target_language = ui_t(language, "target_simplified_chinese")
            target_language = st.text_input(ui_t(language, "target_language"), value=default_target_language, key="batch_target_language")

        if st.button(ui_t(language, "create_jobs"), key="batch_create_jobs", type="primary", use_container_width=True):
            try:
                result = backend.create_jobs_from_sources(
                    sources_text,
                    title_prefix=ui_t(language, "batch_video_title").replace("{n}", ""),
                    resolution=resolution,
                    source_language=source_language,
                    target_language=target_language,
                )
                st.success(f"{ui_t(language, 'jobs_ready')}: {result['created']}")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    _section_title(st, ui_t(language, "running"))
    with st.container(border=True):
        pid = backend.active_scheduler_pid()
        health = backend.scheduler_safety()
        cols = st.columns(4)
        with cols[0]:
            if st.button(ui_t(language, "start_batch"), key="batch_start", use_container_width=True, disabled=bool(pid)):
                if not backend.active_dashboard()["jobs"]:
                    st.error(ui_t(language, "create_jobs_first"))
                else:
                    result = backend.start_scheduler(preset=preset)
                    if result.get("started"):
                        st.success(f"{ui_t(language, 'scheduler_started')}: {result.get('pid')}")
                    else:
                        st.warning(str(result.get("reason") or result))
                    st.rerun()
        with cols[1]:
            if st.button(ui_t(language, "stop_batch"), key="batch_stop", use_container_width=True, disabled=not bool(pid)):
                backend.stop_scheduler()
                st.success(ui_t(language, "scheduler_stopped"))
                st.rerun()
        with cols[2]:
            if st.button(ui_t(language, "reset_failed"), key="batch_reset_failed", use_container_width=True):
                result = backend.reset_failed(preset=preset)
                st.success(f"{ui_t(language, 'failed_reset_done')}: {result['reset']}")
                st.rerun()
        with cols[3]:
            if st.button(ui_t(language, "recover_interrupted"), key="batch_recover", use_container_width=True):
                result = backend.recover()
                st.success(f"{ui_t(language, 'recovered_jobs')}: {result['recovered']}")
                st.rerun()

        if health.get("needs_recovery"):
            st.warning(ui_t(language, "scheduler_recovery_needed"))
        elif pid:
            st.info(f"{ui_t(language, 'scheduler_running')}: {pid}")
        elif health.get("stale_lock"):
            st.warning(ui_t(language, "scheduler_lock_stale"))

        _dependency_health_panel(st, backend, language)
        _render_batch_status(st, backend, language, preset)


def _render_batch_status(st, backend: WebUiBackend, language: str, preset: str) -> None:
    dashboard = backend.active_dashboard()
    jobs = dashboard["jobs"]
    if not jobs:
        st.info(ui_t(language, "no_batch_jobs"))
        return

    done = dashboard["counts"].get("done", 0)
    failed = dashboard["counts"].get("failed", 0)
    running = dashboard["counts"].get("running", 0)
    st.progress(done / len(jobs), text=f"{done}/{len(jobs)} {ui_t(language, 'done')}, {running} {ui_t(language, 'active')}, {failed} {ui_t(language, 'failed')}")

    _resource_lane_panel(st, jobs, language)
    _batch_job_cards(st, jobs, language)
    _stage_matrix_panel(st, jobs, language)

    visible_rows = [
        {
            ui_t(language, "job"): row["job"],
            ui_t(language, "status"): _display_status(row["status"], language),
            ui_t(language, "stage"): _display_stage(row["stage"], language),
            ui_t(language, "progress"): row["progress"],
            ui_t(language, "updated"): row["updated"],
            ui_t(language, "caption_source"): _caption_source_label(str(row.get("caption_source") or ""), language),
            ui_t(language, "error_summary"): str(row.get("error_summary") or ""),
            ui_t(language, "title_or_source"): row["source"],
        }
        for row in jobs
    ]
    st.dataframe(visible_rows, use_container_width=True, hide_index=True)

    with st.expander(ui_t(language, "finished_outputs"), expanded=False):
        has_outputs = False
        for row in jobs:
            detail = backend.job_detail(row["job"])
            if not _stage_done(detail, FULL_TARGET_STAGE):
                continue
            st.markdown(f"**{html.escape(row['job'])}**")
            _render_outputs(st, detail["outputs"], language, kinds={"video", "audio", "subtitle", "data"}, title=None)
            has_outputs = True
        if not has_outputs:
            st.info(ui_t(language, "no_batch_jobs"))

    with st.expander(ui_t(language, "stage_reports"), expanded=False):
        for row in jobs:
            detail = backend.job_detail(row["job"])
            st.markdown(f"**{html.escape(row['job'])}**")
            if detail.get("manifest"):
                st.caption("manifest.json")
                st.json(detail["manifest"])
            if detail.get("quality_report"):
                st.caption("quality_report.json")
                st.json(detail["quality_report"])

    error_rows = [row for row in jobs if row.get("error")]
    if error_rows:
        with st.expander(ui_t(language, "job_messages"), expanded=True):
            for row in error_rows:
                st.write(f"{row['job']}: {row['error']}")


def _history_tab(st, backend: WebUiBackend, language: str) -> None:
    _section_title(st, ui_t(language, "history"), ui_t(language, "history_caption"))
    with st.container(border=True):
        dashboard = backend.history_dashboard()
        jobs = dashboard["jobs"]
        if not jobs:
            st.info(ui_t(language, "no_history_jobs"))
            return

        st.dataframe(
            [
                {
                    ui_t(language, "history_job"): row["job"],
                    ui_t(language, "status"): _display_status(row["status"], language),
                    ui_t(language, "updated"): row["updated"],
                    ui_t(language, "title_or_source"): row["source"],
                }
                for row in jobs
            ],
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(ui_t(language, "history_outputs"), expanded=False):
            for row in jobs:
                detail = backend.job_detail(row["job"])
                st.markdown(f"**{html.escape(row['job'])}**")
                _render_outputs(st, detail["outputs"], language, kinds={"video", "audio", "subtitle"}, title=None)


def _resource_lane_panel(st, rows: list[dict[str, Any]], language: str) -> None:
    running_counts = {stage.value: 0 for stage in STAGE_ORDER}
    for row in rows:
        if row.get("status") == JobStatus.RUNNING.value and row.get("stage") in running_counts:
            running_counts[str(row["stage"])] += 1
    st.markdown(f"**{ui_t(language, 'pipeline_lanes')}**")
    cols = st.columns(len(STAGE_ORDER))
    for column, stage in zip(cols, STAGE_ORDER):
        column.metric(_display_stage(stage.value, language), running_counts[stage.value])


def _batch_job_cards(st, rows: list[dict[str, Any]], language: str) -> None:
    cards = []
    for row in rows:
        status = str(row.get("status") or "pending")
        stage = str(row.get("stage") or "-")
        error_summary = str(row.get("error_summary") or "").strip()
        error_html = (
            f'<div class="eistara-job-error">{html.escape(error_summary)}</div>'
            if error_summary
            else ""
        )
        cards.append(
            f"""
            <div class="eistara-job-card">
              <div class="eistara-job-topline">
                <div class="eistara-job-title">{html.escape(str(row.get("job") or ""))}</div>
                <span class="eistara-pill eistara-pill-{_status_class(status)}">{html.escape(_display_status(status, language))}</span>
              </div>
              <div class="eistara-job-meta">
                <div>{html.escape(ui_t(language, "stage"))}: {html.escape(_display_stage(stage, language))}</div>
                <div>{html.escape(ui_t(language, "progress"))}: {html.escape(str(row.get("progress") or ""))}</div>
                <div>{html.escape(ui_t(language, "caption_source"))}: {html.escape(_caption_source_label(str(row.get("caption_source") or ""), language))}</div>
                <div>{html.escape(ui_t(language, "updated"))}: {html.escape(str(row.get("updated") or ""))}</div>
              </div>
              <div class="eistara-job-source">{html.escape(str(row.get("source") or ""))}</div>
              {error_html}
            </div>
            """
        )
    st.markdown(f'<div class="eistara-job-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _stage_matrix_panel(st, rows: list[dict[str, Any]], language: str) -> None:
    matrix_rows = []
    for row in rows:
        stage_statuses = row.get("stage_statuses") if isinstance(row.get("stage_statuses"), dict) else {}
        item = {
            ui_t(language, "job"): row["job"],
            ui_t(language, "title_or_source"): row["source"],
            ui_t(language, "caption_source"): _caption_source_label(str(row.get("caption_source") or ""), language),
        }
        for stage in STAGE_ORDER:
            item[_display_stage(stage.value, language)] = _stage_status_label(str(stage_statuses.get(stage.value) or "pending"), language)
        matrix_rows.append(item)
    with st.expander(ui_t(language, "stage_matrix"), expanded=False):
        st.dataframe(matrix_rows, use_container_width=True, hide_index=True)


def _dependency_health_panel(st, backend: WebUiBackend, language: str) -> None:
    try:
        health = backend.health()
    except Exception as exc:
        st.warning(str(exc))
        return
    tts_checks = [check for check in health.get("runtime", {}).get("checks", []) if check.get("kind") == "tts"]
    if not tts_checks:
        return
    down = [check for check in tts_checks if not check.get("ok")]
    if down:
        details = "; ".join(f"{check.get('name')}: {check.get('detail')}" for check in down)
        st.warning(f"IndexTTS/TTS service is DOWN. Dubbing stages will wait: {details}")
    else:
        st.success("IndexTTS/TTS service is reachable.")


def _render_header(st, language: str) -> None:
    st.markdown(
        f"""
        <div class="eistara-hero">
          <div>
            <div class="eistara-hero-title">Eistara</div>
            <div class="eistara-hero-subtitle">{html.escape(ui_t(language, "app_subtitle"))}</div>
          </div>
          <div class="eistara-hero-actions">
            <div class="eistara-hero-badge">{html.escape(ui_t(language, "hero_publish"))}</div>
            <div class="eistara-hero-badge">{html.escape(ui_t(language, "batch_ready"))}</div>
            <div class="eistara-hero-badge">{html.escape(ui_t(language, "hero_cover"))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _section_title(st, title: str, caption: str = "") -> None:
    st.markdown(
        f"""
        <div class="eistara-section-title">{html.escape(title)}</div>
        <div class="eistara-section-caption">{html.escape(caption)}</div>
        """,
        unsafe_allow_html=True,
    )


def _steps_html(steps: list[str]) -> str:
    items = "".join(f'<li class="eistara-step">{index}. {html.escape(step)}</li>' for index, step in enumerate(steps, start=1))
    return f'<ul class="eistara-step-list">{items}</ul>'


def _render_job_progress(st, detail: dict[str, Any], language: str) -> None:
    state = detail["state"]
    completed = state.get("completed_stages") or []
    st.progress(len(completed) / len(STAGE_ORDER), text=f"{len(completed)}/{len(STAGE_ORDER)}")
    st.caption(f"{ui_t(language, 'status')}: {state.get('status')} | {ui_t(language, 'stage')}: {state.get('current_stage') or state.get('failed_stage') or '-'}")
    if state.get("error"):
        st.error(_short_ui_error(state["error"]))


def _render_outputs(
    st,
    outputs: list[dict[str, object]],
    language: str,
    *,
    kinds: set[str],
    preview_role: str | None = None,
    title: str | None = None,
) -> None:
    visible = [item for item in outputs if item.get("kind") in kinds and item.get("exists")]
    if not visible:
        return
    if title:
        st.markdown(f"**{html.escape(title)}**")
    preview = next((item for item in visible if item.get("role") == preview_role), None)
    if preview and st.checkbox(ui_t(language, "preview_final_video") if preview_role == "dub_video" else ui_t(language, "preview_source_video"), value=False, key=f"preview-{preview['role']}-{preview['filename']}"):
        st.video(str(preview["path"]))
    cards = []
    for item in visible:
        cards.append(
            f"""
            <div class="eistara-artifact-card">
              <div class="eistara-artifact-topline">
                <span class="eistara-artifact-role">{html.escape(str(item.get("role") or ""))}</span>
                <span class="eistara-artifact-kind">{html.escape(str(item.get("kind") or ""))}</span>
              </div>
              <div class="eistara-artifact-file">{html.escape(str(item.get("filename") or ""))}</div>
              <div class="eistara-artifact-size">{html.escape(_format_size(int(item.get("size") or 0)))}</div>
            </div>
            """
        )
    st.markdown(f'<div class="eistara-artifact-grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    with st.expander(ui_t(language, "artifact_path"), expanded=False):
        st.dataframe(
            [
                {
                    ui_t(language, "artifact_role"): item["role"],
                    ui_t(language, "artifact_file"): item["filename"],
                    ui_t(language, "artifact_kind"): item["kind"],
                    ui_t(language, "artifact_size"): _format_size(int(item.get("size") or 0)),
                    ui_t(language, "artifact_path"): item["path"],
                }
                for item in visible
            ],
            use_container_width=True,
            hide_index=True,
    )


def _stage_done(detail: dict[str, Any], stage: str) -> bool:
    completed = set((detail.get("state") or {}).get("completed_stages") or [])
    return stage in completed


def _config_get(data: dict[str, Any], key: str, default=None):
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _config_text(st, backend: WebUiBackend, config: dict[str, Any], key: str, label: str, *, password: bool = False) -> None:
    current = _config_get(config, key, "")
    value = st.text_input(label, value=str(current or ""), type="password" if password else "default", key=f"config_{key.replace('.', '_')}")
    if value != str(current or ""):
        backend.update_config({key: value})
        st.rerun()


def _config_toggle(st, backend: WebUiBackend, config: dict[str, Any], key: str, label: str) -> None:
    current = bool(_config_get(config, key, False))
    value = st.toggle(label, value=current, key=f"config_{key.replace('.', '_')}")
    if value != current:
        backend.update_config({key: bool(value)})
        st.rerun()


def _config_select(
    st,
    backend: WebUiBackend,
    config: dict[str, Any],
    key: str,
    label: str,
    options: list[str],
    *,
    format_func=None,
) -> None:
    current = str(_config_get(config, key, options[0]) or options[0])
    index = options.index(current) if current in options else 0
    value = st.selectbox(label, options, index=index, key=f"config_{key.replace('.', '_')}", format_func=format_func)
    if value != current:
        backend.update_config({key: value})
        st.rerun()


def _config_number(st, backend: WebUiBackend, config: dict[str, Any], key: str, label: str, minimum: int, maximum: int, *, step: int = 1) -> None:
    current = int(_config_get(config, key, minimum) or minimum)
    value = int(st.number_input(label, min_value=minimum, max_value=maximum, value=current, step=step, key=f"config_{key.replace('.', '_')}"))
    if value != current:
        backend.update_config({key: value})
        st.rerun()


def _config_float_slider(
    st,
    backend: WebUiBackend,
    config: dict[str, Any],
    key: str,
    label: str,
    minimum: float,
    maximum: float,
    *,
    step: float,
) -> None:
    current = float(_config_get(config, key, minimum) or minimum)
    value = float(st.slider(label, min_value=minimum, max_value=maximum, value=current, step=step, key=f"config_{key.replace('.', '_')}"))
    if value != current:
        backend.update_config({key: value})
        st.rerun()


def _recognition_language_options(language: str) -> dict[str, str]:
    if language == "zh-CN":
        return {
            "en": "英语",
            "zh": "简体中文",
            "es": "西班牙语",
            "ru": "俄语",
            "fr": "法语",
            "de": "德语",
            "it": "意大利语",
            "ja": "日语",
            "auto": "自动检测",
        }
    return {
        "en": "English",
        "zh": "Simplified Chinese",
        "es": "Spanish",
        "ru": "Russian",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "ja": "Japanese",
        "auto": "Auto detect",
    }


def _source_language_options(language: str) -> dict[str, str]:
    return {
        ui_t(language, "source_language_english"): "en",
        ui_t(language, "source_language_auto"): "auto",
        ui_t(language, "source_language_chinese"): "zh",
    }


def _batch_config_summary(config: dict[str, Any], language: str) -> str:
    batch = _config_get(config, "batch", {}) or {}
    parts = [f"max={batch.get('max_active_jobs', 10)}"]
    for stage in STAGE_ORDER:
        parts.append(f"{_display_stage(stage.value, language)}={batch.get(stage.value + '_workers', '-')}")
    return " / ".join(parts)


def _display_status(value: str, language: str) -> str:
    if language != "zh-CN":
        return value
    return {
        "pending": "待处理",
        "running": "运行中",
        "done": "完成",
        "failed": "失败",
    }.get(value, value)


def _display_stage(value: str, language: str) -> str:
    if language != "zh-CN":
        return value
    return {
        "download": "下载",
        "transcribe": "转写",
        "translate": "翻译",
        "tts_prepare": "配音准备",
        "tts": "TTS",
        "audio_mix": "混音",
        "compose": "合成",
        "-": "-",
    }.get(value, value)


def _stage_status_label(value: str, language: str) -> str:
    if language != "zh-CN":
        return {
            "pending": "Pending",
            "running": "Running",
            "done": "Done",
            "skipped": "Skipped",
            "failed": "Failed",
        }.get(value, value)
    return {
        "pending": "待",
        "running": "运行",
        "done": "完成",
        "skipped": "跳过",
        "failed": "失败",
    }.get(value, value)


def _caption_source_label(value: str, language: str) -> str:
    if language != "zh-CN":
        return {
            "youtube_subtitle": "YouTube subtitle",
            "existing_cleaned_chunks": "Existing transcript",
        }.get(value, value or "-")
    return {
        "youtube_subtitle": "YouTube字幕",
        "existing_cleaned_chunks": "已有转写",
    }.get(value, value or "-")


def _location_label(value: str, language: str) -> str:
    if language == "zh-CN":
        return {
            "jobs": "进行中",
            "history": "历史",
        }.get(value, value or "-")
    labels = {
        "jobs": "Active",
        "history": "History",
    }
    return labels.get(value, value or "-")


def _status_class(value: str) -> str:
    if value in {"done", "running", "failed"}:
        return value
    return "pending"


def _format_size(size: int) -> str:
    if size <= 0:
        return "-"
    units = ("B", "KB", "MB", "GB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


STYLE = """
<style>
:root {
    --eistara-ink: #101820;
    --eistara-muted: #5f6f7d;
    --eistara-line: #dbe4ea;
    --eistara-soft: #f6f8f8;
    --eistara-accent: #16866f;
    --eistara-accent-soft: #e8f6f1;
    --eistara-blue: #245b8f;
    --eistara-blue-soft: #eaf2fb;
    --eistara-gold: #9a6a18;
    --eistara-gold-soft: #fbf4e4;
    --eistara-error: #b23838;
}
.stApp { background: #f7f9fa; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none; }
[data-testid="stSidebar"] { min-width: 360px; max-width: 360px; }
.block-container { padding-top: 1.05rem; padding-bottom: 3rem; max-width: 1280px; }
h1, h2, h3, label, p, span { letter-spacing: 0 !important; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--eistara-line);
    border-radius: 8px;
    box-shadow: 0 8px 20px rgba(16, 24, 32, 0.035);
    background: #ffffff;
}
div[data-testid="stTabs"] [role="tablist"] {
    gap: 0.35rem;
    border-bottom: 1px solid var(--eistara-line);
}
div[data-testid="stTabs"] button[role="tab"] {
    font-weight: 680;
    border-radius: 7px 7px 0 0;
    padding: 0.45rem 0.9rem;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--eistara-accent);
    background: #ffffff;
}
div.stButton > button, div.stDownloadButton > button {
    border-radius: 7px;
    border: 1px solid #cfd9df;
    background: #ffffff;
    color: var(--eistara-ink);
    font-weight: 620;
    min-height: 2.4rem;
    padding: 0.35rem 0.85rem;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    border-color: var(--eistara-accent);
    color: var(--eistara-accent);
    background: var(--eistara-accent-soft);
}
div.stButton > button[kind="primary"] {
    background: var(--eistara-ink);
    border-color: var(--eistara-ink);
    color: #ffffff;
}
.sidebar-brand {
    padding: 0.25rem 0 1rem 0;
    border-bottom: 1px solid rgba(49, 63, 80, 0.14);
    margin-bottom: 1rem;
}
.sidebar-brand-title {
    font-size: 1.35rem;
    font-weight: 750;
    color: #101820;
    line-height: 1.2;
}
.sidebar-brand-subtitle {
    color: #5d6b78;
    font-size: 0.88rem;
    margin-top: 0.25rem;
}
.eistara-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.05rem 1.2rem;
    background: #ffffff;
    color: var(--eistara-ink);
    border: 1px solid var(--eistara-line);
    border-left: 4px solid var(--eistara-accent);
    border-radius: 8px;
    margin-bottom: 1rem;
    box-shadow: 0 10px 24px rgba(16, 24, 32, 0.055);
}
.eistara-hero-title { font-size: 1.68rem; font-weight: 780; line-height: 1.15; }
.eistara-hero-subtitle { margin-top: 0.35rem; color: var(--eistara-muted); font-size: 0.95rem; }
.eistara-hero-actions {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.45rem;
    max-width: 45%;
}
.eistara-hero-badge {
    color: var(--eistara-accent);
    background: var(--eistara-accent-soft);
    border: 1px solid rgba(22, 134, 111, 0.22);
    padding: 0.35rem 0.65rem;
    border-radius: 999px;
    font-size: 0.86rem;
    font-weight: 680;
    white-space: nowrap;
}
.eistara-hero-badge:nth-child(2) {
    color: var(--eistara-blue);
    background: var(--eistara-blue-soft);
    border-color: rgba(36, 91, 143, 0.2);
}
.eistara-hero-badge:nth-child(3) {
    color: var(--eistara-gold);
    background: var(--eistara-gold-soft);
    border-color: rgba(154, 106, 24, 0.2);
}
.eistara-section-title {
    font-size: 1.04rem;
    font-weight: 720;
    color: var(--eistara-ink);
    margin: 0.15rem 0 0.12rem 0;
}
.eistara-section-caption {
    color: var(--eistara-muted);
    font-size: 0.9rem;
    margin-bottom: 0.65rem;
}
.eistara-mode-note {
    border: 1px solid var(--eistara-line);
    border-left: 3px solid var(--eistara-accent);
    background: #ffffff;
    color: #2f4150;
    border-radius: 8px;
    padding: 0.8rem 0.9rem;
    margin: 0.25rem 0 0.95rem 0;
    font-size: 0.92rem;
    line-height: 1.55;
}
.eistara-mode-card {
    min-height: 12.5rem;
    border: 1px solid var(--eistara-line);
    border-radius: 8px;
    background: #ffffff;
    padding: 1rem 1.05rem;
    margin-bottom: 0.55rem;
    box-shadow: 0 8px 20px rgba(16, 24, 32, 0.04);
}
.eistara-mode-card.subtitle { border-left: 4px solid var(--eistara-blue); }
.eistara-mode-card.dubbing { border-left: 4px solid var(--eistara-accent); }
.eistara-mode-kicker {
    color: var(--eistara-muted);
    font-size: 0.78rem;
    font-weight: 720;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.eistara-mode-title {
    color: var(--eistara-ink);
    font-size: 1.24rem;
    font-weight: 760;
    line-height: 1.25;
    margin-bottom: 0.6rem;
}
.eistara-mode-body {
    color: #354653;
    font-size: 0.94rem;
    line-height: 1.55;
}
.eistara-mode-current {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    border: 1px solid var(--eistara-line);
    background: #ffffff;
    border-radius: 8px;
    padding: 0.62rem 0.75rem;
    margin-bottom: 0.9rem;
    color: var(--eistara-muted);
    font-size: 0.9rem;
}
.eistara-mode-current strong {
    color: var(--eistara-ink);
    font-weight: 740;
}
.eistara-step-list {
    margin: 0.2rem 0 0.85rem 0;
    padding: 0;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.45rem;
}
.eistara-step {
    list-style: none;
    padding: 0.55rem 0.7rem;
    border: 1px solid var(--eistara-line);
    border-left: 3px solid var(--eistara-accent);
    border-radius: 7px;
    background: var(--eistara-soft);
    color: #33414d;
    font-size: 0.9rem;
    min-height: 2.55rem;
    display: flex;
    align-items: center;
}
.eistara-flow-strip {
    display: grid;
    grid-template-columns: minmax(120px, 0.32fr) minmax(0, 1fr);
    align-items: center;
    gap: 0.75rem;
    border: 1px solid var(--eistara-line);
    background: #ffffff;
    border-radius: 8px;
    padding: 0.7rem 0.85rem;
    margin: 0 0 0.9rem 0;
    box-shadow: 0 6px 16px rgba(16, 24, 32, 0.04);
}
.eistara-flow-job {
    color: var(--eistara-muted);
    font-size: 0.86rem;
    font-weight: 680;
    overflow-wrap: anywhere;
}
.eistara-flow-steps {
    display: grid;
    grid-template-columns: repeat(7, minmax(0, 1fr));
    gap: 0.35rem;
}
.eistara-flow-step {
    min-height: 2.1rem;
    border-radius: 7px;
    border: 1px solid var(--eistara-line);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--eistara-muted);
    background: #f8fafb;
    font-size: 0.82rem;
    font-weight: 680;
    text-align: center;
    padding: 0.25rem;
}
.eistara-flow-done {
    color: var(--eistara-accent);
    background: var(--eistara-accent-soft);
    border-color: rgba(22, 134, 111, 0.22);
}
.eistara-flow-current {
    color: var(--eistara-blue);
    background: var(--eistara-blue-soft);
    border-color: rgba(36, 91, 143, 0.22);
}
.eistara-flow-failed {
    color: var(--eistara-error);
    background: #faecec;
    border-color: #f0caca;
}
.eistara-artifact-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 0.55rem;
    margin: 0.35rem 0 0.45rem 0;
}
.eistara-artifact-card {
    border: 1px solid var(--eistara-line);
    border-radius: 8px;
    background: #fbfcfd;
    padding: 0.65rem 0.72rem;
}
.eistara-artifact-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.35rem;
}
.eistara-artifact-role {
    color: var(--eistara-ink);
    font-size: 0.82rem;
    font-weight: 740;
    overflow-wrap: anywhere;
}
.eistara-artifact-kind {
    color: var(--eistara-accent);
    background: var(--eistara-accent-soft);
    border-radius: 999px;
    padding: 0.08rem 0.42rem;
    font-size: 0.74rem;
    font-weight: 700;
    white-space: nowrap;
}
.eistara-artifact-file {
    color: #31414e;
    font-size: 0.86rem;
    line-height: 1.35;
    overflow-wrap: anywhere;
}
.eistara-artifact-size {
    color: var(--eistara-muted);
    font-size: 0.78rem;
    margin-top: 0.25rem;
}
.eistara-job-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 0.65rem;
    margin: 0.55rem 0 0.8rem 0;
}
.eistara-job-card {
    border: 1px solid var(--eistara-line);
    border-radius: 8px;
    background: #ffffff;
    padding: 0.75rem 0.85rem;
    box-shadow: 0 5px 14px rgba(16, 24, 32, 0.045);
}
.eistara-job-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.65rem;
    margin-bottom: 0.45rem;
}
.eistara-job-title { font-weight: 720; color: var(--eistara-ink); overflow-wrap: anywhere; }
.eistara-pill {
    display: inline-flex;
    align-items: center;
    min-height: 1.35rem;
    padding: 0.1rem 0.45rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 680;
    white-space: nowrap;
}
.eistara-pill-done { color: var(--eistara-accent); background: var(--eistara-accent-soft); }
.eistara-pill-running { color: #265d8f; background: #e8f2fb; }
.eistara-pill-failed { color: var(--eistara-error); background: #faecec; }
.eistara-pill-pending { color: #5c6770; background: #eef2f5; }
.eistara-job-meta {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.35rem 0.7rem;
    color: var(--eistara-muted);
    font-size: 0.84rem;
}
.eistara-job-source {
    margin-top: 0.45rem;
    color: #3b4a55;
    font-size: 0.86rem;
    overflow-wrap: anywhere;
}
.eistara-job-error {
    margin-top: 0.5rem;
    color: var(--eistara-error);
    background: #fff3f3;
    border: 1px solid #f0caca;
    border-radius: 7px;
    padding: 0.45rem 0.55rem;
    font-size: 0.82rem;
    line-height: 1.35;
    overflow-wrap: anywhere;
}
.eistara-inline-summary {
    border: 1px solid var(--eistara-line);
    border-left: 3px solid var(--eistara-blue);
    background: #fbfcfd;
    color: #2f4150;
    border-radius: 7px;
    padding: 0.65rem 0.75rem;
    margin: 0.2rem 0 0.9rem 0;
    font-size: 0.88rem;
    line-height: 1.45;
    overflow-wrap: anywhere;
}
div[data-testid="stDataFrame"] {
    border: 1px solid var(--eistara-line);
    border-radius: 8px;
    overflow: hidden;
}
div[data-testid="stAlert"] { border-radius: 8px; }
@media (max-width: 760px) {
    .eistara-hero {
        align-items: flex-start;
        flex-direction: column;
    }
    .eistara-hero-actions {
        justify-content: flex-start;
        max-width: 100%;
    }
    .eistara-job-meta {
        grid-template-columns: 1fr;
    }
    .eistara-flow-strip {
        grid-template-columns: 1fr;
    }
    .eistara-flow-steps {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .eistara-mode-card {
        min-height: auto;
    }
    .eistara-mode-current {
        align-items: flex-start;
        flex-direction: column;
    }
}
</style>
"""


if __name__ == "__main__":
    main()
