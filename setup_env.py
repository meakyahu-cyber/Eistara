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
SUPPORTED_PYTHON_MIN = (3, 10)
SUPPORTED_PYTHON_MAX = (3, 12)


def main() -> int:
    args = parse_args()
    ensure_root()
    ensure_config()
    if not args.skip_venv:
        create_venv(args)
    if not PYTHON.exists():
        raise SystemExit(f"Python in virtualenv was not found: {PYTHON}")
    ensure_venv_python_version()

    check_host_runtime()

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


def create_venv(args: argparse.Namespace) -> None:
    if PYTHON.exists():
        print(f"Using existing virtualenv: {VENV_DIR}")
        return
    print(f"Creating virtualenv: {VENV_DIR}")
    interpreter = find_supported_python()
    if interpreter:
        run([*interpreter, "-m", "venv", str(VENV_DIR)], [])
        return
    create_venv_with_uv(args)


def find_supported_python() -> list[str] | None:
    candidates: list[list[str]] = []
    if is_supported_python(sys.executable):
        candidates.append([sys.executable])
    if os.name == "nt":
        candidates.extend([["py", "-3.10"], ["py", "-3.11"]])
    candidates.extend([["python3.10"], ["python3.11"], ["python"]])
    for candidate in candidates:
        if command_is_supported_python(candidate):
            return candidate
    return None


def create_venv_with_uv(args: argparse.Namespace) -> None:
    uv_command = ensure_uv(args)
    env = os.environ.copy()
    env.setdefault("UV_LINK_MODE", "copy")
    run([*uv_command, "venv", "--python", "3.10", str(VENV_DIR)], [], env=env)


def command_is_supported_python(command: list[str]) -> bool:
    try:
        code = (
            "import sys; "
            f"raise SystemExit(0 if {SUPPORTED_PYTHON_MIN!r} <= sys.version_info[:2] < {SUPPORTED_PYTHON_MAX!r} else 1)"
        )
        return subprocess.run([*command, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def is_supported_python(executable: str) -> bool:
    try:
        code = (
            "import sys; "
            f"raise SystemExit(0 if {SUPPORTED_PYTHON_MIN!r} <= sys.version_info[:2] < {SUPPORTED_PYTHON_MAX!r} else 1)"
        )
        return subprocess.run([executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def ensure_venv_python_version() -> None:
    if command_is_supported_python([str(PYTHON)]):
        return
    raise SystemExit(
        f"Existing virtualenv uses an unsupported Python: {PYTHON}. "
        "Remove .venv and rerun setup_env.py with Python 3.10/3.11 available."
    )


def ensure_uv(args: argparse.Namespace) -> list[str]:
    if command_exists(["uv", "--version"]):
        print("Python 3.10/3.11 was not found; using uv to provision Python 3.10.")
        return ["uv"]
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("Python 3.10/3.11 was not found; using uv to provision Python 3.10.")
        return [sys.executable, "-m", "uv"]
    except (OSError, subprocess.CalledProcessError):
        pass

    print("Python 3.10/3.11 was not found; installing uv bootstrapper with the current Python.")
    code = run([sys.executable, "-m", "pip", "install", "--user", "uv"], pip_index_args(args), check=False)
    if code != 0 and args.china_mirror and not args.pip_index.strip():
        print("WARNING: PyPI mirror failed while installing uv; retrying with the official PyPI index.")
        run([sys.executable, "-m", "pip", "install", "--user", "uv"], [], check=True)
    elif code != 0:
        raise SystemExit(code)
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit("uv installation failed. Install Python 3.10 manually and rerun setup_env.py.") from exc
    return [sys.executable, "-m", "uv"]


def command_exists(command: list[str]) -> bool:
    try:
        return subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def install_dependencies(args: argparse.Namespace) -> None:
    run_pip(["install", "--upgrade", "pip", "setuptools", "wheel"], args, allow_official_fallback=True)
    torch_channel = choose_torch_channel(args.torch)
    run(
        [str(PYTHON), "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", PYTORCH_INDEXES[torch_channel]],
        [],
    )
    run_pip(["install", PROJECT_EXTRAS], args, allow_official_fallback=True)
    run_pip(["install", *ASR_PACKAGES], args, allow_official_fallback=True)

    # Demucs declares torchaudio<2.2 even though the runtime works with modern
    # torch/torchaudio. Install its support packages normally, then install the
    # Demucs wheel without allowing pip to downgrade torch.
    run_pip(["install", *DEMUX_RUNTIME_DEPS], args, allow_official_fallback=True)
    install_demucs_without_deps(args)
    run_pip(["install", *AUDIO_SEPARATOR_PACKAGES], args, allow_official_fallback=True)


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


def check_host_runtime() -> None:
    print("Checking host runtime dependencies...")
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        print("ffmpeg/ffprobe: found")
    else:
        print("WARNING: ffmpeg/ffprobe not found in PATH. Install FFmpeg on the host machine before running Eistara.")
    if shutil.which("nvidia-smi"):
        print("NVIDIA driver: detected")
    else:
        print("NVIDIA driver: not detected. GPU acceleration may be unavailable; CPU mode can still be used.")


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


def run_pip(pip_args: list[str], args: argparse.Namespace, *, allow_official_fallback: bool = False) -> None:
    command = [str(PYTHON), "-m", "pip", *pip_args]
    code = run(command, pip_index_args(args), check=False)
    if code == 0:
        return
    if allow_official_fallback and args.china_mirror and not args.pip_index.strip():
        print("WARNING: PyPI mirror failed; retrying with the official PyPI index.")
        run(command, [], check=True)
        return
    raise SystemExit(code)


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
