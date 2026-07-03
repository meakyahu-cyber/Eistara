# Eistara Release Install

This release package contains the Eistara runtime, WebUI, downloader, ASR,
translation, audio separation, and rendering code. It does not contain or
install the TTS service. Start IndexTTS separately and keep its API URL in
`config.local.yaml` or the WebUI.

## Host Prerequisites

Eistara's normal local dubbing runtime expects a Windows NVIDIA GPU
environment. CPU mode is only a limited debugging fallback, not the recommended
release runtime.

Eistara does not install host-level media/GPU runtimes. Install these first and
keep the default install paths unless you already know you need a custom
layout:

- Python 3.10
- NVIDIA Driver
- CUDA Toolkit 12.8
- CUDNN 9.11.0
- FFmpeg
- IndexTTS service, started separately

Windows FFmpeg example:

```powershell
choco install ffmpeg
```

Windows NVIDIA links:

- [CUDA Toolkit 12.8](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe)
- [CUDNN 9.11.0](https://developer.download.nvidia.com/compute/cudnn/9.11.0/local_installers/cudnn_9.11.0_windows.exe)

After installing, close old terminals, open a new PowerShell window, and run:

```powershell
python --version
ffmpeg -version
ffprobe -version
nvidia-smi
nvcc --version
where cudnn64_9.dll
```

If those commands print versions or a CUDNN path, continue to `python
setup_env.py`.

Only troubleshoot `Path` if one of the checks fails:

- If `nvcc --version` fails, reopen PowerShell or reinstall CUDA Toolkit 12.8.
  If it still fails, add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`
  to system `Path`.
- If `where cudnn64_9.dll` fails, add the CUDNN `bin` folder to system `Path`.
  With the default CUDNN installer, this is usually
  `C:\Program Files\NVIDIA\CUDNN\v9.11\bin\12.9`.
- If `ffmpeg -version` fails after a manual FFmpeg zip install, add that
  FFmpeg `bin` folder to system `Path`.

You normally do not need to edit `CUDA_PATH`; the CUDA Toolkit installer writes
it automatically.

If Python 3.10 is not available, `setup_env.py` will try to install and use
`uv` to provision Python 3.10 for Eistara's `.venv`.

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
- create `config.local.yaml` from `config.example.yaml` when missing.

It will not install FFmpeg, CUDA Toolkit, CUDNN, IndexTTS, or any TTS model.

## Default Runtime Policy

`config.example.yaml` is the template copied to `config.local.yaml`. The
release default enables the source-window dubbing timeline, including gap
borrowing, local audio speed fitting, and IndexTTS adaptive source-window
duration-control retry for clips that still need help before final mixing. The
second retime tier remains disabled by default.

Large cross-stage handoff data is stored under `output/internal`, while
`state.json` keeps compact counts and JSON paths. This is expected: later
stages recover TTS input from `output/internal/tts_segments.json` when inline
arrays are omitted from job state.

Active jobs live under `jobs`. Archived jobs live under `history/<video title>`;
the archive root contains user-facing deliverables, while `work` keeps the full
recoverable job tree.

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
