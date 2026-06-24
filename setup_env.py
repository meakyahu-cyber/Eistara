"""Create the Eistara release runtime environment.

This installer is intended for a copied/release Eistara directory. It creates
``.venv``, installs Python dependencies, prepares non-TTS local models, and
leaves IndexTTS as an external service.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import venv
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

PYPI_CHINA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
HF_CHINA_ENDPOINT = "https://hf-mirror.com"
PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu128": "https://download.pytorch.org/whl/cu128",
}

TORCH_PACKAGES = ["torch==2.8.0", "torchaudio==2.8.0", "torchvision==0.23.0"]
PROJECT_EXTRAS = ".[webui]"
ASR_PACKAGES = [
    "whisperx==3.8.6",
    "faster-whisper==1.2.1",
    "huggingface-hub>=0.36,<1.0",
    "librosa>=0.11.0",
]
DEMUX_RUNTIME_DEPS = [
    "dora-search",
    "einops",
    "julius>=0.2.3",
    "lameenc>=1.2",
    "openunmix",
    "tqdm",
]
DEMUX_PACKAGES = ["demucs==4.1.0a3", "demucs==4.0.1"]
AUDIO_SEPARATOR_PACKAGES = ["audio-separator==0.44.2", "onnxruntime-gpu==1.23.2"]

DEFAULT_ASR_MODEL = "Systran/faster-whisper-large-v3"
ZH_ASR_MODEL = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
UVR_MODEL_NAME = "UVR-MDX-NET-Voc_FT.onnx"
UVR_MODEL_SIZE = 66762490
UVR_MODEL_MD5 = "d21dc03e4b9ef397b47231f483af6db8"
UVR_MODEL_URL = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
UVR_MODEL_MIRROR_URL = "https://gh.llkk.cc/https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"


def main() -> int:
    args = parse_args()
    ensure_root()
    ensure_config()
    if not args.skip_venv:
        create_venv()
    if not PYTHON.exists():
        raise SystemExit(f"Python in virtualenv was not found: {PYTHON}")

    if args.install_ffmpeg:
        install_portable_ffmpeg()
    else:
        check_ffmpeg_hint()

    if not args.skip_deps:
        install_dependencies(args)

    if not args.skip_models:
        prepare_models(args)

    run_health_summary()
    print()
    print("Eistara environment is ready.")
    print("Start it with: start_eistara.bat")
    print("WebUI: http://localhost:10127")
    print("TTS is not installed by this script. Start your IndexTTS service separately.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install Eistara release dependencies and non-TTS models.")
    parser.add_argument("--skip-venv", action="store_true", help="Do not create .venv.")
    parser.add_argument("--skip-deps", action="store_true", help="Do not install Python packages.")
    parser.add_argument("--skip-models", action="store_true", help="Do not download or prepare model caches.")
    parser.add_argument("--china-mirror", action=argparse.BooleanOptionalAction, default=True, help="Use China-friendly PyPI/HF mirrors when possible.")
    parser.add_argument("--pip-index", default="", help="Override pip index URL. Defaults to Tsinghua mirror when --china-mirror is enabled.")
    parser.add_argument("--hf-endpoint", default="", help="Override HuggingFace endpoint. Defaults to hf-mirror.com when --china-mirror is enabled.")
    parser.add_argument("--torch", choices=["auto", "cpu", "cu128"], default="auto", help="PyTorch wheel channel.")
    parser.add_argument("--install-ffmpeg", action="store_true", help="Download a portable ffmpeg into tools/ffmpeg.")
    parser.add_argument("--with-zh-asr", action="store_true", help="Also cache the Chinese Belle faster-whisper model.")
    parser.add_argument("--skip-demucs-model", action="store_true", help="Do not prefetch the Demucs htdemucs checkpoint.")
    parser.add_argument("--skip-uvr-mdx", action="store_true", help="Do not download the optional UVR-MDX audio-separator model.")
    return parser.parse_args()


def ensure_root() -> None:
    required = [ROOT / "pyproject.toml", ROOT / "eistara", ROOT / "apps" / "webui"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("This script must be run from an Eistara release root. Missing: " + ", ".join(missing))


def ensure_config() -> None:
    local_config = ROOT / "config.local.yaml"
    template = ROOT / "config.example.yaml"
    if local_config.exists():
        return
    if template.exists():
        shutil.copyfile(template, local_config)
    else:
        local_config.write_text(
            "api:\n"
            "  key: \"\"\n"
            "  base_url: \"\"\n"
            "  model: \"\"\n"
            "indextts:\n"
            "  api_url: \"http://127.0.0.1:8010/tts\"\n",
            encoding="utf-8",
        )
    print("Created config.local.yaml. Fill LLM settings in WebUI or edit the file later.")


def create_venv() -> None:
    if PYTHON.exists():
        print(f"Using existing virtualenv: {VENV_DIR}")
        return
    print(f"Creating virtualenv: {VENV_DIR}")
    venv.EnvBuilder(with_pip=True, clear=False).create(VENV_DIR)


def install_dependencies(args: argparse.Namespace) -> None:
    run([str(PYTHON), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], pip_index_args(args))
    torch_channel = choose_torch_channel(args.torch)
    run(
        [str(PYTHON), "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", PYTORCH_INDEXES[torch_channel]],
        [],
    )
    run([str(PYTHON), "-m", "pip", "install", PROJECT_EXTRAS], pip_index_args(args))
    run([str(PYTHON), "-m", "pip", "install", *ASR_PACKAGES], pip_index_args(args))

    # Demucs declares torchaudio<2.2 even though the runtime works with modern
    # torch/torchaudio. Install its support packages normally, then install the
    # Demucs wheel without allowing pip to downgrade torch.
    run([str(PYTHON), "-m", "pip", "install", *DEMUX_RUNTIME_DEPS], pip_index_args(args))
    install_demucs_without_deps(args)
    run([str(PYTHON), "-m", "pip", "install", *AUDIO_SEPARATOR_PACKAGES], pip_index_args(args))


def install_demucs_without_deps(args: argparse.Namespace) -> None:
    last_error = 0
    for package in DEMUX_PACKAGES:
        code = run(
            [str(PYTHON), "-m", "pip", "install", package, "--no-deps"],
            pip_index_args(args),
            check=False,
        )
        if code == 0:
            return
        last_error = code
    raise SystemExit(f"Demucs installation failed with exit code {last_error}.")


def prepare_models(args: argparse.Namespace) -> None:
    model_dir = ROOT / "_model_cache"
    model_dir.mkdir(exist_ok=True)
    hf_endpoint = args.hf_endpoint.strip() or (HF_CHINA_ENDPOINT if args.china_mirror else "")
    download_hf_snapshot(DEFAULT_ASR_MODEL, model_dir, hf_endpoint)
    if args.with_zh_asr:
        download_hf_snapshot(ZH_ASR_MODEL, model_dir, hf_endpoint)
    if not args.skip_demucs_model:
        prefetch_demucs_model()
    if not args.skip_uvr_mdx:
        download_uvr_model(args.china_mirror)


def download_hf_snapshot(repo_id: str, cache_dir: Path, hf_endpoint: str) -> None:
    print(f"Preparing HuggingFace model: {repo_id}")
    code = (
        "from huggingface_hub import snapshot_download; "
        f"snapshot_download(repo_id={repo_id!r}, cache_dir={str(cache_dir)!r}, resume_download=True)"
    )
    env = os.environ.copy()
    if hf_endpoint:
        env["HF_ENDPOINT"] = hf_endpoint
    run([str(PYTHON), "-c", code], [], env=env)


def prefetch_demucs_model() -> None:
    print("Preparing Demucs checkpoint: htdemucs")
    code = "from demucs.pretrained import get_model; get_model('htdemucs'); print('Demucs htdemucs ready')"
    run([str(PYTHON), "-c", code], [])


def download_uvr_model(china_mirror: bool) -> None:
    target_dir = ROOT / "models" / "audio-separator"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / UVR_MODEL_NAME
    if valid_file(target, UVR_MODEL_SIZE, UVR_MODEL_MD5):
        print(f"UVR-MDX model already ready: {target}")
        return
    url = UVR_MODEL_MIRROR_URL if china_mirror else UVR_MODEL_URL
    partial = target.with_suffix(target.suffix + ".part")
    if partial.exists():
        partial.unlink()
    print(f"Downloading UVR-MDX model: {url}")
    urlretrieve(url, partial)
    if not valid_file(partial, UVR_MODEL_SIZE, UVR_MODEL_MD5):
        partial.unlink(missing_ok=True)
        raise SystemExit("Downloaded UVR-MDX model failed integrity check.")
    partial.replace(target)
    print(f"UVR-MDX model ready: {target}")


def valid_file(path: Path, expected_size: int, expected_md5: str) -> bool:
    if not path.exists() or path.stat().st_size != expected_size:
        return False
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_md5.lower()


def install_portable_ffmpeg() -> None:
    bin_dir = ROOT / "tools" / "ffmpeg" / "bin"
    ffmpeg = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    ffprobe = bin_dir / ("ffprobe.exe" if os.name == "nt" else "ffprobe")
    if ffmpeg.exists() and ffprobe.exists():
        print(f"Portable ffmpeg already ready: {bin_dir}")
        return

    download_dir = ROOT / "_downloads"
    archive = download_dir / "ffmpeg-release-essentials.zip"
    extract_dir = ROOT / "tools" / "ffmpeg" / "_extract"
    download_dir.mkdir(exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ffmpeg: {FFMPEG_URL}")
    urlretrieve(FFMPEG_URL, archive)

    if extract_dir.resolve().is_relative_to((ROOT / "tools" / "ffmpeg").resolve()):
        shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(extract_dir)
    candidates = list(extract_dir.glob("**/bin/ffmpeg.exe"))
    if not candidates:
        raise SystemExit("ffmpeg.exe was not found in downloaded archive.")
    source_bin = candidates[0].parent
    bin_dir.mkdir(parents=True, exist_ok=True)
    for item in source_bin.iterdir():
        if item.is_file():
            shutil.copy2(item, bin_dir / item.name)
    shutil.rmtree(extract_dir, ignore_errors=True)
    print(f"Portable ffmpeg ready: {bin_dir}")


def check_ffmpeg_hint() -> None:
    local_bin = ROOT / "tools" / "ffmpeg" / "bin"
    if shutil.which("ffmpeg") or (local_bin / "ffmpeg.exe").exists():
        return
    print("WARNING: ffmpeg was not found. Re-run with --install-ffmpeg or install ffmpeg manually.")


def run_health_summary() -> None:
    print("Checking installed runtime packages...")
    code = (
        "import importlib.metadata as m; "
        "pkgs=['streamlit','torch','torchaudio','whisperx','faster-whisper','demucs','audio-separator','onnxruntime-gpu','yt-dlp']; "
        "missing=[]; "
        "\nfor p in pkgs:\n"
        "    try: print(f'{p}: {m.version(p)}')\n"
        "    except m.PackageNotFoundError: missing.append(p)\n"
        "\nprint('missing: ' + ', '.join(missing) if missing else 'missing: none')"
    )
    run([str(PYTHON), "-c", code], [], check=False)


def choose_torch_channel(value: str) -> str:
    if value != "auto":
        return value
    return "cu128" if shutil.which("nvidia-smi") else "cpu"


def pip_index_args(args: argparse.Namespace) -> list[str]:
    index = args.pip_index.strip() or (PYPI_CHINA_INDEX if args.china_mirror else "")
    if not index:
        return []
    result = ["--index-url", index]
    host = urlparse(index).hostname
    if host:
        result.extend(["--trusted-host", host])
    return result


def run(cmd: list[str], extra_args: list[str], *, env: dict[str, str] | None = None, check: bool = True) -> int:
    full_cmd = [*cmd, *extra_args]
    print("+ " + " ".join(quote(part) for part in full_cmd))
    proc = subprocess.run(full_cmd, cwd=str(ROOT), env=env)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc.returncode


def quote(value: str) -> str:
    if not value or any(ch.isspace() for ch in value):
        return f'"{value}"'
    return value


if __name__ == "__main__":
    raise SystemExit(main())
