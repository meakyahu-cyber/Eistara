# Eistara Release Install

This release package contains the Eistara runtime, WebUI, downloader, ASR,
translation, audio separation, and rendering code. It does not contain or
install the TTS service. Start IndexTTS separately and keep its API URL in
`config.local.yaml` or the WebUI.

## Host Prerequisites

Install [CUDA Toolkit 12.8](https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_571.96_windows.exe), [CUDNN 9.11.0](https://developer.download.nvidia.com/compute/cudnn/9.11.0/local_installers/cudnn_9.11.0_windows.exe), and FFmpeg.

Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin` to the system `Path`.

After installing FFmpeg, make sure FFmpeg is also available from the system `Path`.

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
