from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE_ROOT = ROOT.parent / "Eistara_releases"
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".research_github",
    "_downloads",
    "_model_cache",
    "batch",
    "build",
    "dist",
    "history",
    "jobs",
    "logs",
    "models",
    "output",
    "scripts",
    "tests",
    "work",
}
EXCLUDED_ANY_DIRS = {
    ".pytest_cache",
    "__pycache__",
}
EXCLUDED_FILES = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "README.zh.md",
    "config.local.yaml",
    ".env",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".log",
}
TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".ps1",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = (
    re.compile("s" + r"k-[A-Za-z0-9_\-]{20,}"),
)


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_root = Path(args.release_root).expanduser().resolve()
    package_name = f"Eistara_release_{timestamp}"
    staging = release_root / package_name
    archive = release_root / f"{package_name}.zip"
    manifest = release_root / f"{package_name}.manifest.json"

    release_root.mkdir(parents=True, exist_ok=True)
    if staging.exists() or archive.exists() or manifest.exists():
        raise SystemExit("Release target already exists; rerun to get a fresh timestamp.")

    copy_release_tree(staging)
    scan_for_secrets(staging)
    records = build_manifest(staging)
    write_zip(staging, archive)
    manifest.write_text(
        json.dumps(
            {
                "package": package_name,
                "source": str(ROOT),
                "staging": str(staging),
                "archive": str(archive),
                "files": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Release staging: {staging}")
    print(f"Release archive: {archive}")
    print(f"Manifest: {manifest}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an Eistara release package.")
    parser.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT), help="Directory that receives release folders and zips.")
    return parser.parse_args()


def copy_release_tree(target: Path) -> None:
    target.mkdir(parents=True)
    for source in ROOT.rglob("*"):
        relative = source.relative_to(ROOT)
        if should_exclude(relative, source):
            continue
        destination = target / relative
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def should_exclude(relative: Path, source: Path) -> bool:
    parts = relative.parts
    if parts and parts[0] in EXCLUDED_DIRS:
        return True
    if set(parts) & EXCLUDED_ANY_DIRS:
        return True
    if any(part.endswith(".egg-info") for part in parts):
        return True
    if source.is_file() and source.name in EXCLUDED_FILES:
        return True
    if source.is_file() and source.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return False


def scan_for_secrets(root: Path) -> None:
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            hits.append(str(path.relative_to(root)))
    if hits:
        raise SystemExit("Potential API key found in release staging: " + ", ".join(hits))


def build_manifest(root: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        records.append(
            {
                "path": str(path.relative_to(root)).replace(os.sep, "/"),
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(source: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            zf.write(path, path.relative_to(source.parent))


if __name__ == "__main__":
    raise SystemExit(main())
