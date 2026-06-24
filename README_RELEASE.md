# Eistara Release Install

This release package contains the Eistara runtime, WebUI, downloader, ASR,
translation, audio separation, and rendering code. It does not contain or
install the TTS service. Start IndexTTS separately and keep its API URL in
`config.local.yaml` or the WebUI.

## Host Prerequisites

Eistara does not install host-level media/GPU runtimes. Install these on the
machine first:

- Python 3.10 or 3.11
- FFmpeg and FFprobe, available in `PATH`
- NVIDIA driver for GPU acceleration
- CUDA Toolkit / CUDNN only when your local GPU stack explicitly needs them
- IndexTTS service, started separately

Windows FFmpeg example:

```powershell
choco install ffmpeg
```

Windows NVIDIA links:

- [CUDA Toolkit 12.6](https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.76_windows.exe)
- [CUDNN 9.3.0](https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn_9.3.0_windows.exe)

If Python 3.10/3.11 is not available, `setup_env.py` will try to install and
use `uv` to provision Python 3.10 for Eistara's `.venv`.

## One-command Install

Open PowerShell in the release directory and run:

```powershell
python setup_env.py
```

The installer will:

- create `.venv`;
- install Python dependencies into `.venv`;
- install PyTorch, WhisperX, Demucs, yt-dlp, and WebUI deps;
- cache the default `Systran/faster-whisper-large-v3` ASR model;
- prefetch the Demucs `htdemucs` checkpoint;
- download the optional UVR-MDX model used by the audio-separator backend;
- create `config.local.yaml` from `config.example.yaml` when missing.

It will not install FFmpeg, CUDA Toolkit, CUDNN, IndexTTS, or any TTS model.

## Useful Options

```powershell
python setup_env.py --help
python setup_env.py --torch cpu
python setup_env.py --torch cu128
python setup_env.py --with-zh-asr
python setup_env.py --skip-models
python setup_env.py --no-china-mirror
```

By default, pip uses the Tsinghua PyPI mirror and HuggingFace downloads use
`hf-mirror.com`. PyTorch wheels still use the official PyTorch wheel index.

## Start

```powershell
start_eistara.bat
```

WebUI runs at:

```text
http://localhost:10127
```

Fill LLM `base_url`, `key`, and `model` in the WebUI. The WebUI writes those
settings to `config.local.yaml`; the rest of Eistara uses that same config.
