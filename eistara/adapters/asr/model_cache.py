from __future__ import annotations

from pathlib import Path


def resolve_faster_whisper_model(model: str, model_dir: str | Path, *, local_alias: str | None = None) -> str:
    """Return a local model path when Eistara can find cached weights."""
    model_text = str(model or "").strip()
    model_cache_dir = Path(model_dir).expanduser()
    candidate = Path(model_text).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return str(candidate)

    for name in (local_alias, model_text):
        if not name:
            continue
        local_candidate = model_cache_dir / name
        if _has_faster_whisper_weights(local_candidate):
            return str(local_candidate)

    snapshot = _cached_huggingface_snapshot(_repo_id_for_faster_whisper(model_text), model_cache_dir)
    if snapshot is not None:
        return str(snapshot)
    return model_text


def faster_whisper_cache_path(model: str, model_dir: str | Path) -> Path:
    model_text = str(model or "").strip()
    model_cache_dir = Path(model_dir).expanduser()
    candidate = Path(model_text).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    local_candidate = model_cache_dir / model_text
    if local_candidate.exists():
        return local_candidate
    repo_id = _repo_id_for_faster_whisper(model_text)
    snapshot = _cached_huggingface_snapshot(repo_id, model_cache_dir)
    if snapshot is not None:
        return snapshot
    return _huggingface_cache_root(repo_id, model_cache_dir)


def has_faster_whisper_weights(path: str | Path) -> bool:
    return _has_faster_whisper_weights(Path(path))


def _repo_id_for_faster_whisper(model: str) -> str:
    return model if "/" in model else f"Systran/faster-whisper-{model}"


def _huggingface_cache_root(repo_id: str, model_dir: Path) -> Path:
    return model_dir / ("models--" + repo_id.replace("/", "--"))


def _cached_huggingface_snapshot(repo_id: str, model_dir: Path) -> Path | None:
    cache_root = _huggingface_cache_root(repo_id, model_dir)
    refs_main = cache_root / "refs" / "main"
    try:
        ref = refs_main.read_text(encoding="utf-8").strip()
    except OSError:
        ref = ""
    if ref:
        snapshot = cache_root / "snapshots" / ref
        if _has_faster_whisper_weights(snapshot):
            return snapshot

    snapshots_dir = cache_root / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshots = [path for path in snapshots_dir.iterdir() if _has_faster_whisper_weights(path)]
    if not snapshots:
        return None
    return max(snapshots, key=lambda path: path.stat().st_mtime)


def _has_faster_whisper_weights(path: Path) -> bool:
    if path.is_file():
        return path.name in {"model.bin", "model.safetensors"}
    if not path.exists():
        return False
    return any(path.glob("**/model.bin")) or any(path.glob("**/model.safetensors"))
