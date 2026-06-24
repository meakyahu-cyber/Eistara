"""Eistara launcher - pre-flight checks and logging."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = SCRIPT_DIR / "logs"
LOG_FILE = LOG_DIR / f"startup_{datetime.now():%Y%m%d_%H%M%S}.log"
STREAMLIT_PORT = int(os.environ.get("EISTARA_STREAMLIT_PORT", "10127"))


def configure_local_tools() -> None:
    """Prefer portable tools bundled by the release installer when present."""

    local_ffmpeg_bin = SCRIPT_DIR / "tools" / "ffmpeg" / "bin"
    ffmpeg_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    if (local_ffmpeg_bin / ffmpeg_name).exists():
        os.environ["PATH"] = str(local_ffmpeg_bin) + os.pathsep + os.environ.get("PATH", "")


def log(msg: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def bootstrap_config() -> None:
    """Create the local config from a tracked template when one is available."""

    local_config = SCRIPT_DIR / "config.local.yaml"
    legacy_config = SCRIPT_DIR / "config.yaml"
    template_file = SCRIPT_DIR / "config.example.yaml"
    if local_config.exists() or legacy_config.exists() or not template_file.exists():
        return
    shutil.copyfile(template_file, local_config)
    log(f"Created {local_config.name} from {template_file.name}")
    print()
    print("  Created config.local.yaml from config.example.yaml.")
    print("  Set EISTARA_API_KEY in your environment, or edit config.local.yaml before use.")
    print()


def check_package(name: str, import_name: str | None = None) -> str | None:
    import_name = import_name or name
    if importlib.util.find_spec(import_name) is None:
        return None
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "installed"


def streamlit_is_responding() -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{STREAMLIT_PORT}", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []

    configure_local_tools()
    bootstrap_config()

    log(f"Python: {sys.version.split()[0]} ({sys.executable})")

    for pkg, imp in [("streamlit", None), ("json_repair", "json_repair")]:
        if not check_package(pkg, imp):
            errors.append(f"{pkg} not installed. Install dependencies in .venv before launching Eistara.")

    torch_ver = check_package("torch")
    if torch_ver:
        log(f"torch: {torch_ver} (CUDA check deferred)")

    if not check_package("whisperx"):
        warnings.append("whisperx not installed. ASR will fail if the local WhisperX provider is selected.")

    if not shutil.which("ffmpeg"):
        errors.append("ffmpeg not found in PATH. Install ffmpeg or add it to PATH.")

    port_in_use = False
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", STREAMLIT_PORT)) == 0:
            port_in_use = True
            if streamlit_is_responding():
                log(f"Eistara already responding on port {STREAMLIT_PORT}")
                print()
                print(f"  Eistara is already running: http://localhost:{STREAMLIT_PORT}")
                print()
                return
            warnings.append(
                f"Port {STREAMLIT_PORT} in use by another process. Close it or set EISTARA_STREAMLIT_PORT."
            )

    for warning in warnings:
        log(f"[WARN] {warning}")
    for error in errors:
        log(f"[ERROR] {error}")

    if errors:
        print()
        for error in errors:
            print(f"  [ERROR] {error}")
        print(f"\n  Fix errors above. Log: {LOG_FILE}\n")
        sys.exit(1)
    if port_in_use:
        print(f"\n  Port {STREAMLIT_PORT} is occupied by another process. See: {LOG_FILE}\n")
        sys.exit(1)
    if warnings:
        print()
        for warning in warnings:
            print(f"  [WARN] {warning}")
        print()

    log("Launching Streamlit...")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "st.py",
                "--server.port",
                str(STREAMLIT_PORT),
                "--logger.level",
                "error",
            ],
            cwd=str(SCRIPT_DIR),
        )
        if proc.returncode != 0:
            log(f"Streamlit exited with code {proc.returncode}")
            print(f"\n  Streamlit crashed (code {proc.returncode}). See: {LOG_FILE}\n")
            sys.exit(proc.returncode)
    except KeyboardInterrupt:
        log("Stopped by user")


if __name__ == "__main__":
    main()
