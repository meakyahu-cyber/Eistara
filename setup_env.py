"""Create the Eistara release runtime environment.

This installer is intended for a copied/release Eistara directory. It creates
``.venv``, installs Python dependencies, prepares non-TTS local models, and
leaves IndexTTS as an external service.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
PYTHON = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

PYPI_CHINA_INDEX = "https://pypi.tuna.tsinghua.edu.cn/simple"
HF_CHINA_ENDPOINT = "https://hf-mirror.com"
PYTORCH_INDEXES = {
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu129": "https://download.pytorch.org/whl/cu129",
}

TORCH_PACKAGE_VERSIONS = {
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "torchvision": "0.23.0",
}
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
DEMUX_PACKAGE = "demucs @ git+https://github.com/adefossez/demucs@b9ab48cad45976ba42b2ff17b229c071f0df9390"

DEFAULT_ASR_MODEL = "Systran/faster-whisper-large-v3"
SUPPORTED_PYTHON_VERSION = (3, 10)


def main() -> int:
    args = parse_args()
    ensure_root()
    ensure_config()
    if not args.skip_venv:
        create_venv(args)
    if not PYTHON.exists():
        raise SystemExit(f"Python in virtualenv was not found: {PYTHON}")
    ensure_venv_python_version()
    if not args.skip_deps:
        ensure_pip()

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
    parser.add_argument("--torch", choices=["auto", "cpu", "cu126", "cu128", "cu129"], default="auto", help="PyTorch wheel channel.")
    parser.add_argument("--skip-demucs-model", action="store_true", help="Do not prefetch the Demucs htdemucs checkpoint.")
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
        candidates.extend([["py", "-3.10"]])
    candidates.extend([["python3.10"], ["python"]])
    for candidate in candidates:
        if command_is_supported_python(candidate):
            return candidate
    return None


def create_venv_with_uv(args: argparse.Namespace) -> None:
    uv_command = ensure_uv(args)
    env = os.environ.copy()
    env.setdefault("UV_LINK_MODE", "copy")
    run([*uv_command, "venv", "--seed", "--python", "3.10", str(VENV_DIR)], [], env=env)


def command_is_supported_python(command: list[str]) -> bool:
    try:
        code = (
            "import sys; "
            f"raise SystemExit(0 if sys.version_info[:2] == {SUPPORTED_PYTHON_VERSION!r} else 1)"
        )
        return subprocess.run([*command, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def is_supported_python(executable: str) -> bool:
    try:
        code = (
            "import sys; "
            f"raise SystemExit(0 if sys.version_info[:2] == {SUPPORTED_PYTHON_VERSION!r} else 1)"
        )
        return subprocess.run([executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False).returncode == 0
    except OSError:
        return False


def ensure_venv_python_version() -> None:
    if command_is_supported_python([str(PYTHON)]):
        return
    raise SystemExit(
        f"Existing virtualenv uses an unsupported Python: {PYTHON}. "
        "Remove .venv and rerun setup_env.py with Python 3.10 available."
    )


def ensure_pip() -> None:
    if run([str(PYTHON), "-m", "pip", "--version"], [], check=False) == 0:
        return
    print("pip was not found in .venv; bootstrapping with ensurepip.")
    run([str(PYTHON), "-m", "ensurepip", "--upgrade"], [])


def ensure_uv(args: argparse.Namespace) -> list[str]:
    if command_exists(["uv", "--version"]):
        print("Python 3.10 was not found; using uv to provision Python 3.10.")
        return ["uv"]
    try:
        subprocess.run([sys.executable, "-m", "uv", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print("Python 3.10 was not found; using uv to provision Python 3.10.")
        return [sys.executable, "-m", "uv"]
    except (OSError, subprocess.CalledProcessError):
        pass

    print("Python 3.10 was not found; installing uv bootstrapper with the current Python.")
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
    install_torch_packages(args)
    run_pip(["install", PROJECT_EXTRAS], args, allow_official_fallback=True)
    run_pip(["install", *ASR_PACKAGES], args, allow_official_fallback=True)

    # Demucs declares torchaudio<2.2 even though the runtime works with modern
    # torch/torchaudio. Install its support packages normally, then install the
    # Demucs wheel without allowing pip to downgrade torch.
    run_pip(["install", *DEMUX_RUNTIME_DEPS], args, allow_official_fallback=True)
    install_demucs_without_deps(args)


def install_demucs_without_deps(args: argparse.Namespace) -> None:
    code = run([str(PYTHON), "-m", "pip", "install", DEMUX_PACKAGE, "--no-deps"], [], check=False)
    if code == 0:
        return
    raise SystemExit(
        "Demucs 4.1.0a3 installation failed. Eistara pins Demucs to the same "
        "Git commit used by the development environment; check GitHub/network access and rerun setup_env.py."
    )


def prepare_models(args: argparse.Namespace) -> None:
    model_dir = ROOT / "_model_cache"
    model_dir.mkdir(exist_ok=True)
    hf_endpoint = args.hf_endpoint.strip() or (HF_CHINA_ENDPOINT if args.china_mirror else "")
    download_hf_snapshot(DEFAULT_ASR_MODEL, model_dir, hf_endpoint)
    if not args.skip_demucs_model:
        prefetch_demucs_model()


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
    if shutil.which("nvcc"):
        print("CUDA Toolkit: detected")
    else:
        print(
            "WARNING: CUDA Toolkit nvcc not found. "
            "Install CUDA Toolkit 12.8 with default settings, then open a new PowerShell."
        )
    cudnn_path = find_file_on_path("cudnn64_9.dll")
    if cudnn_path:
        print(f"CUDNN: detected ({cudnn_path.parent})")
    else:
        print(
            "WARNING: CUDNN cudnn64_9.dll not found. "
            "Install CUDNN 9.11.0 with default settings, then open a new PowerShell."
        )


def find_file_on_path(filename: str) -> Path | None:
    for raw_entry in os.environ.get("PATH", "").split(os.pathsep):
        entry = raw_entry.strip().strip('"')
        if not entry:
            continue
        candidate = Path(entry) / filename
        if candidate.exists():
            return candidate
    return None


def run_health_summary() -> None:
    print("Checking installed runtime packages...")
    code = (
        "import importlib.metadata as m; "
        "pkgs=['streamlit','torch','torchaudio','torchvision','whisperx','faster-whisper','demucs','yt-dlp']; "
        "missing=[]; "
        "\nfor p in pkgs:\n"
        "    try: print(f'{p}: {m.version(p)}')\n"
        "    except m.PackageNotFoundError: missing.append(p)\n"
        "\nprint('missing: ' + ', '.join(missing) if missing else 'missing: none')"
    )
    run([str(PYTHON), "-c", code], [], check=False)


def install_torch_packages(args: argparse.Namespace) -> None:
    channels = torch_channel_candidates(args.torch)
    existing = installed_torch_channel()
    if args.torch == "auto" and existing in channels:
        print(f"PyTorch packages already match a compatible channel: {existing}")
        return
    if existing:
        print(f"Existing PyTorch channel is {existing}; installing {channels[0]} packages.")

    failures: list[str] = []
    for index, channel in enumerate(channels):
        specs = torch_package_specs(channel)
        print(torch_install_message(channel))
        code = run_torch_install(channel, specs, args)
        if code == 0:
            return
        failures.append(f"{channel}: exit {code}")
        if index + 1 < len(channels):
            print(f"WARNING: PyTorch channel {channel} failed; retrying with {channels[index + 1]}.")
    raise SystemExit(
        "PyTorch installation failed ("
        + "; ".join(failures)
        + "). Check network access or rerun with --torch cpu."
    )


def torch_channel_candidates(value: str) -> list[str]:
    if value != "auto":
        return [value]
    if not shutil.which("nvidia-smi"):
        return ["cpu"]
    driver_cuda = detect_nvidia_driver_cuda_version()
    if driver_cuda is None:
        print("WARNING: Could not parse NVIDIA driver CUDA version; trying cu126.")
        return ["cu126"]
    if driver_cuda >= (12, 9):
        return ["cu129", "cu128", "cu126"]
    if driver_cuda >= (12, 8):
        return ["cu128", "cu126"]
    if driver_cuda >= (12, 6):
        return ["cu126"]
    version_text = f"{driver_cuda[0]}.{driver_cuda[1]}"
    raise SystemExit(
        "NVIDIA driver CUDA capability is too old for the supported CUDA PyTorch wheels "
        f"({version_text}). Update the NVIDIA driver or rerun setup_env.py with --torch cpu."
    )


def detect_nvidia_driver_cuda_version() -> tuple[int, int] | None:
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    output = f"{result.stdout}\n{result.stderr}"
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", output)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def torch_package_specs(channel: str) -> list[str]:
    if channel == "cpu":
        return [f"{package}=={version}" for package, version in TORCH_PACKAGE_VERSIONS.items()]
    return [f"{package}=={version}+{channel}" for package, version in TORCH_PACKAGE_VERSIONS.items()]


def torch_install_message(channel: str) -> str:
    if channel == "cpu":
        return "Installing PyTorch channel: cpu (configured PyPI index)"
    return f"Installing PyTorch channel: {channel} ({PYTORCH_INDEXES[channel]})"


def run_torch_install(channel: str, specs: list[str], args: argparse.Namespace) -> int:
    command = [str(PYTHON), "-m", "pip", "install", *specs]
    if channel == "cpu":
        code = run(command, pip_index_args(args), check=False)
        if code == 0:
            return 0
        if args.china_mirror and not args.pip_index.strip():
            print("WARNING: PyPI mirror failed while installing CPU PyTorch; retrying with the official PyPI index.")
            return run(command, [], check=False)
        return code
    return run([*command, "--index-url", PYTORCH_INDEXES[channel]], [], check=False)


def installed_torch_channel() -> str | None:
    versions = installed_torch_versions()
    if not versions or any(not versions.get(package) for package in TORCH_PACKAGE_VERSIONS):
        return None
    channels: set[str] = set()
    for package, expected_version in TORCH_PACKAGE_VERSIONS.items():
        version = str(versions.get(package) or "")
        base, _, local = version.partition("+")
        if base != expected_version:
            return "outdated"
        channels.add(local or "cpu")
    if len(channels) == 1:
        return channels.pop()
    return "mixed"


def installed_torch_versions() -> dict[str, str] | None:
    code = (
        "import importlib.metadata as m, json; "
        f"pkgs={list(TORCH_PACKAGE_VERSIONS)!r}; "
        "versions={}; "
        "\nfor p in pkgs:\n"
        "    try: versions[p]=m.version(p)\n"
        "    except m.PackageNotFoundError: versions[p]=''\n"
        "\nprint(json.dumps(versions))"
    )
    try:
        result = subprocess.run(
            [str(PYTHON), "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return None
    return {str(key): str(value) for key, value in payload.items()}


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
    print("+ " + " ".join(quote(part) for part in full_cmd), flush=True)
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
