from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.config import AppConfig
from eistara.core.subtitle.nlp_split import DEFAULT_SPACY_MODEL_MAP


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INDEXTTS_CHECKPOINT_HINTS = (
    Path(os.environ.get("EISTARA_INDEXTTS_DIR", "")) if os.environ.get("EISTARA_INDEXTTS_DIR") else None,
    Path(os.environ.get("INDEXTTS_HOME", "")) if os.environ.get("INDEXTTS_HOME") else None,
    Path(r"D:\AI\Index-TTS-V26"),
)
INDEXTTS_REQUIRED_FILES = (
    "dist/index-tts-windows-cu128-deepspeed/data/checkpoints/gpt.pth",
    "dist/index-tts-windows-cu128-deepspeed/data/checkpoints/s2mel.pth",
    "dist/index-tts-windows-cu128-deepspeed/data/checkpoints/qwen0.6bemo4-merge/model.safetensors",
)


@dataclass(frozen=True, slots=True)
class ModelDependencyItem:
    component: str
    dependency: str
    mode: str
    required: bool
    ok: bool | None
    status: str
    detail: str = ""
    path: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "dependency": self.dependency,
            "mode": self.mode,
            "required": self.required,
            "ok": self.ok,
            "status": self.status,
            "detail": self.detail,
            "path": self.path,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class ModelDependencyReport:
    items: tuple[ModelDependencyItem, ...]

    @property
    def ok(self) -> bool:
        return all(item.ok is not False for item in self.items if item.required)

    def to_dict(self) -> dict[str, Any]:
        counts = {
            "required": sum(1 for item in self.items if item.required),
            "missing_required": sum(1 for item in self.items if item.required and item.ok is False),
            "optional_missing": sum(1 for item in self.items if not item.required and item.ok is False),
            "unknown": sum(1 for item in self.items if item.ok is None),
        }
        return {
            "ok": self.ok,
            "counts": counts,
            "items": [item.to_dict() for item in self.items],
        }


def build_model_dependency_report(config: AppConfig, *, project_root: Path | None = None) -> ModelDependencyReport:
    root = (project_root or PROJECT_ROOT).resolve()
    items: list[ModelDependencyItem] = []
    items.extend(_source_items(config))
    items.extend(_llm_items(config))
    items.extend(_tts_items(config))
    items.extend(_asr_items(config, root))
    items.extend(_vocal_separation_items(config, root))
    items.extend(_spacy_items(config))
    return ModelDependencyReport(tuple(items))


def _source_items(config: AppConfig) -> list[ModelDependencyItem]:
    yt_dlp_version = _package_version("yt-dlp")
    executable = str(config.source.yt_dlp_path or "yt-dlp")
    return [
        ModelDependencyItem(
            component="download",
            dependency="YouTube / yt-dlp",
            mode="network",
            required=True,
            ok=bool(yt_dlp_version),
            status="available" if yt_dlp_version else "package missing",
            detail=f"downloads and YouTube subtitle fetches require network; executable={executable}",
            version=yt_dlp_version,
        )
    ]


def _llm_items(config: AppConfig) -> list[ModelDependencyItem]:
    has_base_url = bool(str(config.api.base_url or "").strip())
    has_key = bool(str(config.api.key or "").strip())
    has_model = bool(str(config.api.model or "").strip())
    ok = has_base_url and has_key and has_model
    missing = []
    if not has_base_url:
        missing.append("base_url")
    if not has_key:
        missing.append("key")
    if not has_model:
        missing.append("model")
    return [
        ModelDependencyItem(
            component="translate",
            dependency="LLM gateway",
            mode="network",
            required=True,
            ok=ok,
            status="configured" if ok else "missing " + ", ".join(missing),
            detail=f"model={config.api.model or '-'}; proxy={'env' if config.api.trust_env_proxy else 'direct'}",
            path=str(config.api.base_url or ""),
            version=_package_version("openai"),
        )
    ]


def _tts_items(config: AppConfig) -> list[ModelDependencyItem]:
    method = str(config.tts_method or "").strip() or "indextts"
    provider_config = config.tts_backend_config(method)
    api_url = str(provider_config.get("api_url") or "").strip()
    items = [
        ModelDependencyItem(
            component="tts",
            dependency=f"TTS provider ({method})",
            mode="local-service" if method in {"indextts", "custom_tts", "gpt_sovits"} else "network",
            required=True,
            ok=bool(api_url) if method in {"indextts", "custom_tts"} else None,
            status="configured" if api_url else "external provider config",
            detail="Eistara calls the provider through its configured API; model weights are owned by that service.",
            path=api_url,
        )
    ]
    if method == "indextts":
        items.append(_indextts_checkpoint_item())
    return items


def _asr_items(config: AppConfig, project_root: Path) -> list[ModelDependencyItem]:
    provider = str(config.asr.provider or "local").strip().lower()
    model, model_path = _selected_local_asr_model(config, project_root)
    items: list[ModelDependencyItem] = []
    if provider in {"whisperx", "local", "whisperx-local"}:
        whisperx_version = _package_version("whisperx")
        faster_version = _package_version("faster-whisper")
        model_ok = _has_faster_whisper_weights(model_path)
        items.append(
            ModelDependencyItem(
                component="asr",
                dependency=f"WhisperX local ASR ({model})",
                mode="local-cache",
                required=True,
                ok=bool(whisperx_version and faster_version and model_ok),
                status="cached" if model_ok else "cache missing; would need HuggingFace/hf-mirror",
                detail=f"whisperx={whisperx_version or 'missing'}; faster-whisper={faster_version or 'missing'}",
                path=str(model_path),
                version=faster_version,
            )
        )
        items.append(_alignment_item(config))
        return items
    if provider in {"whisperx-302", "302", "cloud"}:
        return [
            ModelDependencyItem(
                component="asr",
                dependency="WhisperX 302 cloud",
                mode="network",
                required=True,
                ok=bool(str(config.asr.provider_config.get("whisperX_302_api_key") or "").strip()),
                status="configured" if config.asr.provider_config.get("whisperX_302_api_key") else "missing API key",
                path="https://api.302.ai/302/whisperx",
            )
        ]
    if provider == "elevenlabs":
        return [
            ModelDependencyItem(
                component="asr",
                dependency="ElevenLabs speech-to-text",
                mode="network",
                required=True,
                ok=bool(str(config.asr.provider_config.get("elevenlabs_api_key") or "").strip()),
                status="configured" if config.asr.provider_config.get("elevenlabs_api_key") else "missing API key",
                path="https://api.elevenlabs.io/v1/speech-to-text",
            )
        ]
    return [
        ModelDependencyItem(
            component="asr",
            dependency=f"ASR provider ({provider})",
            mode="unknown",
            required=True,
            ok=None,
            status="unknown provider",
        )
    ]


def _vocal_separation_items(config: AppConfig, project_root: Path) -> list[ModelDependencyItem]:
    items: list[ModelDependencyItem] = []
    enabled = bool(config.demucs.enabled)
    demucs_version = _package_version("demucs")
    cache_files = _demucs_cache_files()
    items.append(
        ModelDependencyItem(
            component="vocal_separation",
            dependency="Demucs htdemucs",
            mode="local-cache",
            required=enabled,
            ok=bool(demucs_version and cache_files),
            status="cached" if cache_files else "cache missing; Demucs may download on first use",
            detail=f"{len(cache_files)} candidate checkpoint(s)",
            path="; ".join(str(path) for path in cache_files[:3]),
            version=demucs_version,
        )
    )
    return items


def _spacy_items(config: AppConfig) -> list[ModelDependencyItem]:
    model_map = dict(DEFAULT_SPACY_MODEL_MAP)
    model_map.update(config.runtime.spacy_model_map or {})
    items = [
        ModelDependencyItem(
            component="subtitle_split",
            dependency="spaCy package",
            mode="local-package",
            required=False,
            ok=bool(_package_version("spacy")),
            status="installed" if _package_version("spacy") else "package missing",
            detail="missing language models fall back to spacy.blank()+sentencizer",
            version=_package_version("spacy"),
        )
    ]
    for language, model_name in sorted(model_map.items()):
        installed = importlib.util.find_spec(model_name) is not None
        items.append(
            ModelDependencyItem(
                component="subtitle_split",
                dependency=f"spaCy {language} model ({model_name})",
                mode="local-package",
                required=False,
                ok=installed,
                status="installed" if installed else "fallback blank model",
                detail="optional quality dependency for sentence splitting",
            )
        )
    return items


def _indextts_checkpoint_item() -> ModelDependencyItem:
    root = _first_existing_path(INDEXTTS_CHECKPOINT_HINTS)
    if root is None:
        return ModelDependencyItem(
            component="tts",
            dependency="IndexTTS local checkpoints",
            mode="local-service-model",
            required=False,
            ok=None,
            status="not discoverable from Eistara",
            detail="set EISTARA_INDEXTTS_DIR or INDEXTTS_HOME to let the report inspect the service checkpoints",
        )
    missing = [relative for relative in INDEXTTS_REQUIRED_FILES if not (root / relative).exists()]
    return ModelDependencyItem(
        component="tts",
        dependency="IndexTTS local checkpoints",
        mode="local-service-model",
        required=False,
        ok=not missing,
        status="found" if not missing else "incomplete",
        detail="missing: " + ", ".join(missing) if missing else "core checkpoint files found",
        path=str(root),
    )


def _alignment_item(config: AppConfig) -> ModelDependencyItem:
    language = str(config.asr.language or "en").split("-")[0].lower()
    checkpoint_dir = _torch_checkpoint_dir()
    if language == "en":
        path = checkpoint_dir / "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
        return ModelDependencyItem(
            component="asr",
            dependency="WhisperX English alignment model",
            mode="local-cache",
            required=False,
            ok=path.exists(),
            status="cached" if path.exists() else "cache missing; torchaudio may download",
            path=str(path),
        )
    return ModelDependencyItem(
        component="asr",
        dependency=f"WhisperX {language} alignment model",
        mode="local-cache",
        required=False,
        ok=None,
        status="language-specific cache not inspected",
        detail="non-English alignment models may require HuggingFace unless cached",
    )


def _faster_whisper_cache_path(model: str, model_dir: Path) -> Path:
    candidate = Path(model).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    local_candidate = model_dir / model
    if local_candidate.exists():
        return local_candidate
    repo_id = model if "/" in model else f"Systran/faster-whisper-{model}"
    return model_dir / ("models--" + repo_id.replace("/", "--"))


def _selected_local_asr_model(config: AppConfig, project_root: Path) -> tuple[str, Path]:
    model_dir = _resolve_project_path(config.runtime.model_dir, project_root)
    language = str(config.asr.language or "").strip().lower()
    if language == "zh":
        local_name = "Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_path = model_dir / local_name
        if local_path.exists():
            return local_name, local_path
        repo_id = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        return repo_id, _faster_whisper_cache_path(repo_id, model_dir)
    model = str(config.asr.model or "large-v3").strip()
    return model, _faster_whisper_cache_path(model, model_dir)


def _has_faster_whisper_weights(path: Path) -> bool:
    if path.is_file():
        return path.name in {"model.bin", "model.safetensors"}
    if not path.exists():
        return False
    return any(path.glob("**/model.bin")) or any(path.glob("**/model.safetensors"))


def _demucs_cache_files() -> list[Path]:
    checkpoint_dir = _torch_checkpoint_dir()
    if not checkpoint_dir.exists():
        return []
    candidates = list(checkpoint_dir.glob("955717e8*.th"))
    candidates.extend(path for path in checkpoint_dir.glob("*htdemucs*.th") if path not in candidates)
    return sorted(path for path in candidates if path.is_file())


def _torch_checkpoint_dir() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    root = Path(torch_home).expanduser() if torch_home else Path.home() / ".cache" / "torch"
    return root / "hub" / "checkpoints"


def _resolve_project_path(path: str | os.PathLike[str], project_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _first_existing_path(paths: tuple[Path | None, ...]) -> Path | None:
    for path in paths:
        if path and path.exists():
            return path.resolve()
    return None


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""
