# Eistara

Eistara is a Windows-first video translation and Chinese dubbing workflow. It
handles source acquisition, ASR, LLM translation, Demucs vocal separation,
IndexTTS-based dubbing, subtitle generation, audio mixing, and final video
rendering through the WebUI and CLI.

Chinese install guide: [README.zh.md](README.zh.md)

Release-package install guide: [README_RELEASE.md](README_RELEASE.md)

## Runtime Requirements

- Python 3.10.x
- FFmpeg and FFprobe available in `PATH`
- NVIDIA driver when GPU acceleration is used
- CUDA Toolkit / CUDNN only when your local GPU stack explicitly needs them
- IndexTTS service started separately, default API URL:
  `http://127.0.0.1:8010/tts`

Eistara's installer creates `.venv`, installs Python dependencies, prepares the
non-TTS model cache, and creates `config.local.yaml` from
`config.example.yaml`. It does not install FFmpeg, CUDA Toolkit, CUDNN,
IndexTTS, or TTS model files.

## Install

```powershell
git clone https://github.com/meakyahu-cyber/Eistara.git
cd Eistara
python setup_env.py
```

Useful installer options:

```powershell
python setup_env.py --skip-models
python setup_env.py --torch cpu
python setup_env.py --torch cu128
python setup_env.py --with-zh-asr
python setup_env.py --no-china-mirror
```

By default, pip uses the Tsinghua PyPI mirror, HuggingFace downloads use
`hf-mirror.com`, and PyTorch wheels use the official PyTorch wheel index.

## Start

```powershell
start_eistara.bat
```

The WebUI runs at:

```text
http://localhost:10127
```

Fill LLM `base_url`, `key`, and `model` in the WebUI. These settings are saved
to local `config.local.yaml`, which is not committed.

## Default Runtime Policy

`config.example.yaml` is the template copied to `config.local.yaml`.

- Translation batches default to 20 lines.
- ASR defaults to `Systran/faster-whisper-large-v3`.
- Optional Chinese ASR cache: `Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper`.
- Vocal separation uses Demucs `htdemucs`.
- Dubbing uses the `source_window` timeline by default.
- Source-window retiming enables gap borrowing and local audio speed fitting.
- IndexTTS adaptive source-window duration-control retry is enabled for clips
  that still need help before final mixing.
- The second retime tier remains disabled by default.
- Background audio uses adaptive wideband ducking by default.

## Job Directories

Active jobs live under `jobs/<job_id>`.

Completed jobs are archived under `history/<video title>`. The archive root is
for user-facing deliverables such as the source video, dubbed video, and
subtitle files. The full recoverable working tree is kept under
`history/<video title>/work`.

Large cross-stage handoff data is stored under `output/internal`, while
`state.json` keeps compact counts and JSON paths. Later stages can recover TTS
input from `output/internal/tts_segments.json` when inline arrays were omitted
from persisted job state.

## CLI Checks

The WebUI is the normal entry point, but the CLI can be used for health and
debug checks:

```powershell
python -m apps.cli.main --jobs-dir .\jobs health
python -m apps.cli.main --jobs-dir .\jobs status
python -m apps.cli.main --jobs-dir .\jobs events
python -m apps.cli.main --jobs-dir .\jobs stages
python -m apps.cli.main --config .\config.local.yaml --jobs-dir .\jobs run-once --preset production
```

## Code Layout

- `apps.webui`: Streamlit operations UI.
- `apps.cli`: command-line health, scheduler, delivery, and debug tools.
- `eistara.runtime`: production scheduler assembly and runtime health checks.
- `eistara.config`: defaults, config loading, and typed settings builders.
- `eistara.core.jobs`: job state model, JSON job store, and archive handling.
- `eistara.core.scheduler`: stage scheduling, locks, heartbeat, and recovery.
- `eistara.core.source`: local/URL source acquisition.
- `eistara.core.asr`: ASR request/result models and transcribe runners.
- `eistara.core.translation`: batching, prompting, validation, and publishing.
- `eistara.core.tts`: TTS request models, cache, retry service, and text cleanup.
- `eistara.core.timeline`: source-window timeline preparation.
- `eistara.core.dubbing`: audio placement, retiming, mixing, and rendering flow.
- `eistara.core.delivery`: user-facing video/subtitle deliverables.
- `eistara.adapters`: ASR, TTS, LLM, media, and source adapter boundaries.
- `eistara.core.diagnostics`: optional local diagnostics hooks; disabled by
  default and not part of core workflow behavior.
