# Eistara Release Install

This release package contains the Eistara runtime, WebUI, downloader, ASR,
translation, audio separation, and rendering code. It does not contain or
install the TTS service. Start IndexTTS separately and keep its API URL in
`config.local.yaml` or the WebUI.

## One-command Install

Open PowerShell in the release directory and run:

```powershell
python setup_env.py --install-ffmpeg
```

The installer will:

- create `.venv`;
- install Python dependencies into `.venv`;
- install PyTorch, WhisperX, Demucs, yt-dlp, ffmpeg helpers, and WebUI deps;
- cache the default `Systran/faster-whisper-large-v3` ASR model;
- prefetch the Demucs `htdemucs` checkpoint;
- download the optional UVR-MDX model used by the audio-separator backend;
- create `config.local.yaml` from `config.example.yaml` when missing.

It will not install IndexTTS or any TTS model.

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
