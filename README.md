# Eistara

Eistara is the maintained video translation and dubbing pipeline.
This repository contains the runtime, WebUI, CLI, adapters, and core workflow
modules used for current development.

## Architecture

The codebase is split around stable workflow boundaries:

- `eistara.core.jobs`: job state model and JSON-backed job store.
- `eistara.core.jobs.factory`: task input parsing and isolated job creation.
- `eistara.core.manifest`: manifest model and JSON manifest store.
- `eistara.core.pipeline`: `StageRunner`, `StageContext`, and `StageResult`.
- `eistara.core.pipeline.artifacts`: per-stage output contracts and artifact
  checks.
- `eistara.core.pipeline.registry`: runner registration and missing-stage
  inspection.
- `eistara.core.scheduler`: small scheduler service that mutates state and calls
  registered stage runners.
- `eistara.core.scheduler.lock`: scheduler lock and stale-lock cleanup.
- `eistara.core.scheduler.heartbeat`: scheduler liveness heartbeat.
- `eistara.core.scheduler.recovery`: orphaned scheduler/job recovery.
- `eistara.core.scheduler.status`: structured status and scheduler health.
- `eistara.core.translation`: publish-first translation policy split into
  batching, prompting, validation, fallback/service orchestration, and a
  lightweight stage runner.
- `eistara.core.tts`: TTS policy split into request/result models, text
  normalization, cache signatures, provider protocol, retrying service, and a
  lightweight stage runner.
- `eistara.adapters.tts.indextts`: IndexTTS HTTP adapter. It owns payload
  construction, service readiness probing, HTTP status classification, and
  writing returned audio bytes.
- `eistara.core.asr`: ASR request/result models, segment normalization,
  subtitle row conversion, provider protocol, audio-extract plus ASR glue, and
  lightweight transcribe stage runners.
- `eistara.adapters.asr.whisper`: Whisper and Faster-Whisper adapter boundary.
  Heavy model dependencies stay outside the core package.
- `eistara.adapters.llm.openai_compatible`: OpenAI-compatible chat completion
  adapter for LLM gateways. It owns HTTP payloads, model listing, JSON content
  parsing, and request/service error classification.
- `eistara.core.source`: source acquisition request/result models, local file
  provider, and download stage runner.
- `eistara.adapters.source.ytdlp`: yt-dlp adapter boundary for URL acquisition.
- `eistara.config`: default configuration, lightweight YAML/JSON loading,
  nested key lookup, and typed settings builders for translation/TTS.
- `eistara.core.delivery`: delivery artifact roles, subtitle/video output
  profile, source-video subtitle aliasing, source-timeline subtitle generation,
  and stale subtitle cleanup.
- `eistara.core.subtitle`: pure subtitle primitives for SRT timecodes, visible
  text length, display splitting, subtitle events, and SRT rendering.
- `eistara.core.timeline`: pure dub timeline models, spacing policy, and
  subtitle event generation for `output_dub.srt`, plus TTS-output to
  `internal/dub_segments.json` preparation.
- `eistara.core.dubbing`: audio clip placement plans, dub audio mix plans,
  final render plans, and lightweight stage runners for audio-mix/compose
  planning.
- Pipeline JSON files intended for machine handoff, such as
  `translations.json`, `subtitle_rows.json`, `dub_segments.json`,
  `audio_mix_plan.json`, and `compose_plan.json`, are written under
  `output/internal`; user-facing delivery files stay in `output`.
- `eistara.adapters.media.dubbing_ffmpeg`: ffmpeg-backed renderer for audio
  mix plans and final video composition. Stage runners keep rendering opt-in so
  planning and real subprocess execution stay separate.
- `eistara.core.quality`: unified quality reports, issue severity, translation
  residue checks, subtitle sanity checks, timeline/audio-mix checks, and a
  lightweight quality gate runner.
- `eistara.core.observability`: JSONL job events for stage start, finish,
  retry, failure, outputs, errors, and duration tracking.
- `eistara.runtime.pipeline`: production scheduler assembly. `production` wires
  configured source acquisition, ASR, LLM, TTS, dubbing, and ffmpeg adapters.
- `apps.webui`: minimal Streamlit operations UI for Eistara jobs, health,
  events, quality reports, and safe scheduler actions.
- `eistara.runtime.health`: dependency health checks for runtime tools and
  optional external services such as LLM gateways and TTS servers.
- `eistara.adapters`: boundaries for ASR, TTS, LLM, media, and source adapters.

## Smoke Test

From this directory:

```powershell
python -m apps.cli.main --jobs-dir .\jobs status
python -m apps.cli.main --jobs-dir .\jobs events
python -m apps.cli.main --config .\config.yaml --jobs-dir .\jobs run-once --preset production
python -m apps.cli.main --jobs-dir .\jobs health
python -m apps.cli.main --jobs-dir .\jobs webui --server-port 8501
python -m apps.cli.main --config .\config.yaml --jobs-dir .\jobs health
python -m apps.cli.main --jobs-dir .\jobs health --llm-base-url https://example.com/v1 --tts-api-url http://127.0.0.1:8010/tts
python -m apps.cli.main --jobs-dir .\jobs stages
python -m apps.cli.main delivery list .\work\demo_output
python -m apps.cli.main delivery alias .\work\demo_output .\work\demo_output\source_video.webm
python -m apps.cli.main delivery subtitles .\work\demo_output .\work\subtitle_rows.json
python -m apps.cli.main delivery dub-subtitle .\work\demo_output .\work\dub_segments.json
python -m apps.cli.main asr normalize .\work\asr_segments.json
python -m apps.cli.main llm models --base-url http://127.0.0.1:8000/v1 --model your-model
python -m apps.cli.main dubbing plan .\work\dub_segments.json .\work\source_video.mp4 .\work\demo_output
python -m apps.cli.main render audio-mix .\work\demo_output\internal\audio_mix_plan.json --dry-run
python -m apps.cli.main render compose .\work\source_video.mp4 .\work\demo_output\dub.mp3 .\work\demo_output\output_dub.mp4 --subtitle-path .\work\demo_output\output_dub.srt --dry-run
python -m apps.cli.main quality check --translations-json .\work\demo_output\internal\translations.json --subtitle-rows-json .\work\demo_output\internal\subtitle_rows.json --dub-segments-json .\work\demo_output\internal\dub_segments.json
```

The runtime CLI now exposes the production pipeline only; lightweight scheduler
tests use test-local runners instead of public no-op presets.
