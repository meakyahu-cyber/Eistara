from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from eistara.config import ConfigLoader
from eistara.config.youtube_cookies import apply_youtube_cookie_config, browser_cookie_candidates
from eistara.adapters.llm import OpenAICompatibleLlmClient, OpenAICompatibleSettings
from eistara.adapters.media import FfmpegDubbingRenderer, FfmpegMediaProvider, build_audio_mix_ffmpeg_args
from eistara.core.asr import AsrSegment, asr_segments_to_subtitle_rows, normalize_asr_segments
from eistara.core.delivery import DeliveryPublisher, SubtitleDeliveryGenerator
from eistara.core.dubbing import DubbingRenderService
from eistara.core.jobs import JobFactory, JsonJobStore, JobStatus, StageName, history_dir_for_jobs
from eistara.core.observability import JsonlEventStore
from eistara.core.pipeline import output_internal_path
from eistara.core.quality.runner import _load_subtitle_rows, _load_timeline, _load_translations
from eistara.core.quality import QualityGateService
from eistara.core.scheduler import SchedulerLock, SchedulerProcessTick, SchedulerService, collect_status_rows, recover_orphaned_scheduler_state, scheduler_health
from eistara.runtime import PIPELINE_PRESETS, RuntimeHealthService, build_model_dependency_report, build_process_supervisor, build_scheduler


PROCESS_SCHEDULER_PRESETS = {"production"}


def cmd_status(args: argparse.Namespace) -> int:
    rows = collect_status_rows(args.jobs_dir)
    if not rows:
        print("No jobs found.")
        return 0
    for row in rows:
        print(f"{row['job']}\t{row['status']}\t{row['stage']}\t{row.get('location', 'jobs')}")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    stores = [JsonlEventStore(args.jobs_dir), JsonlEventStore(history_dir_for_jobs(args.jobs_dir))]
    if args.job_id:
        events = []
        for store in stores:
            events.extend(store.read_job(args.job_id))
    else:
        events = []
        for store in stores:
            events.extend(store.read_all())
        events = sorted(events, key=lambda event: event.created_at)
    if args.limit:
        events = events[-int(args.limit) :]
    print(json.dumps([event.to_dict() for event in events], ensure_ascii=False, indent=2))
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    count = recover_orphaned_scheduler_state(args.jobs_dir)
    print(f"Recovered jobs: {count}")
    return 0


def cmd_retry(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    service = build_scheduler(
        args.jobs_dir,
        preset=args.preset,
        config=config,
        max_stage_retries=config.batch.max_stage_retries,
    )
    print(json.dumps(service.retry_failed(args.job_id), ensure_ascii=False, indent=2))
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    service = build_scheduler(
        args.jobs_dir,
        preset=args.preset,
        config=config,
        max_stage_retries=config.batch.max_stage_retries,
    )
    print(json.dumps(service.reset_from_stage(args.job_id, args.stage), ensure_ascii=False, indent=2))
    return 0


def cmd_create(args: argparse.Namespace) -> int:
    created = JobFactory(args.jobs_dir, config_path=args.config).create_from_file(args.input)
    print(f"Created jobs: {len(created)}")
    for path in created:
        print(path.name)
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    llm_base_url = args.llm_base_url if args.llm_base_url is not None else config.api.base_url
    llm_model = args.llm_model if args.llm_model is not None else config.api.model
    tts_api_url = args.tts_api_url
    tts_label = args.tts_label
    if tts_api_url is None and config.tts_method == "indextts":
        tts_api_url = str(config.indextts.get("api_url") or "")
        tts_label = "TTS (indextts)"
    health = {
        "scheduler": scheduler_health(args.jobs_dir),
        "runtime": RuntimeHealthService().check(
            llm_base_url=llm_base_url,
            llm_api_key=config.api.key,
            llm_model=llm_model,
            llm_support_json=config.api.llm_support_json,
            llm_proxy_url=config.api.proxy_url,
            llm_trust_env_proxy=config.api.trust_env_proxy,
            tts_api_url=tts_api_url,
            tts_label=tts_label,
        ).to_dict(),
    }
    print(json.dumps(health, ensure_ascii=False, indent=2))
    return 0


def cmd_dependency_report(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    print(json.dumps(build_model_dependency_report(config).to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_youtube(args: argparse.Namespace) -> int:
    if args.youtube_command == "cookies":
        if args.cookies_action == "detect":
            print(json.dumps([item.to_dict() for item in browser_cookie_candidates(browser_hint=args.browser_hint or "")], ensure_ascii=False, indent=2))
            return 0
        if args.cookies_action == "apply":
            config_path = Path(args.config).expanduser() if args.config else Path("config.yaml")
            result = apply_youtube_cookie_config(
                config_path,
                browser=args.browser,
                profile=args.profile,
                dry_run=args.dry_run,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result.get("updated") or args.dry_run else 1
    raise ValueError(f"Unknown youtube command: {args.youtube_command}")


def cmd_stages(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    service = build_scheduler(Path(args.jobs_dir), preset="production", config=config)
    print("registered:")
    for stage in service.registry.registered_stages():
        print(f"  {stage.value}")
    print("missing:")
    for stage in service.registry.missing_stages():
        print(f"  {stage.value}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    if _use_process_scheduler(args):
        return cmd_run_process(args, config)
    service = build_scheduler(
        args.jobs_dir,
        preset=args.preset,
        config=config,
        render_audio=args.render_audio,
        render_video=args.render_video,
        max_stage_retries=args.max_stage_retries if args.max_stage_retries is not None else config.batch.max_stage_retries,
    )
    if args.command == "run-once":
        ran = service.run_once_with_lock(clear_lock=args.clear_lock)
        print(json.dumps({"ran": ran, "preset": args.preset}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-loop":
        iterations = 0
        ran_count = 0
        while args.max_iterations is None or iterations < args.max_iterations:
            iterations += 1
            ran = service.run_once_with_lock(clear_lock=args.clear_lock and iterations == 1)
            ran_count += 1 if ran else 0
            if not ran and args.stop_when_idle:
                break
            time.sleep(float(args.poll_interval))
        print(json.dumps({"iterations": iterations, "ran_count": ran_count, "preset": args.preset}, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"Unknown run command: {args.command}")


def cmd_run_process(args: argparse.Namespace, config) -> int:
    supervisor = build_process_supervisor(
        args.jobs_dir,
        preset=args.preset,
        config=config,
        config_path=args.config,
        render_audio=args.render_audio,
        render_video=args.render_video,
        max_stage_retries=args.max_stage_retries if args.max_stage_retries is not None else config.batch.max_stage_retries,
    )
    if args.command == "run-once":
        tick = supervisor.run_once_with_lock(clear_lock=args.clear_lock, wait=True)
        print(json.dumps({"mode": "processes", "preset": args.preset, **_process_tick_dict(tick)}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "run-loop":
        iterations = 0
        totals = SchedulerProcessTick()
        try:
            with SchedulerLock(args.jobs_dir, clear_lock=args.clear_lock):
                while args.max_iterations is None or iterations < args.max_iterations:
                    iterations += 1
                    tick = supervisor.run_once(launch=True)
                    totals = totals.merge(tick)
                    if args.stop_when_idle and not supervisor.active and not tick.did_work:
                        idle_action = _classify_process_idle(supervisor.service)
                        if idle_action == "done":
                            break
                        if idle_action == "wait":
                            time.sleep(float(args.poll_interval))
                            continue
                        print("No runnable jobs found. Check failed states or stage limits.")
                        break
                    time.sleep(float(args.poll_interval))
                if supervisor.active:
                    totals = totals.merge(supervisor.wait_for_active(poll_interval=float(args.poll_interval)))
                supervisor.service.heartbeat.clear()
        except KeyboardInterrupt:
            supervisor.terminate_all()
            return 130
        print(
            json.dumps(
                {"mode": "processes", "iterations": iterations, "preset": args.preset, **_process_tick_dict(totals)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise ValueError(f"Unknown run command: {args.command}")


def _use_process_scheduler(args: argparse.Namespace) -> bool:
    if getattr(args, "in_process", False):
        return False
    if getattr(args, "processes", False):
        return True
    return args.command == "run-loop" and str(args.preset) in PROCESS_SCHEDULER_PRESETS


def _classify_process_idle(service: SchedulerService) -> str:
    """Classify process scheduler idle state: done, wait for held/requeue, or stop."""
    jobs = service.job_store.discover()
    unfinished = [job for job in jobs if job.state.status not in {JobStatus.DONE, JobStatus.FAILED}]
    if not unfinished and not _has_pending_requeue(service, jobs):
        return "done"
    if _has_dependency_held_ready_job(service, jobs) or _has_pending_requeue(service, jobs):
        return "wait"
    return "stop"


def _has_pending_requeue(service: SchedulerService, jobs) -> bool:
    recovery = service.recovery
    if not recovery.auto_requeue_failed or recovery.max_auto_requeues <= 0:
        return False
    return any(
        job.state.status == JobStatus.FAILED and job.state.auto_requeue_count < recovery.max_auto_requeues
        for job in jobs
    )


def _has_dependency_held_ready_job(service: SchedulerService, jobs) -> bool:
    registered = set(service.registry.registered_stages())
    for job in jobs:
        if job.state.status != JobStatus.PENDING:
            continue
        stage = service.job_store.next_stage(job.state)
        if stage is None or stage not in registered:
            continue
        ready, _reason = service.dependencies.ready(job, stage)
        if not ready:
            return True
    return False


def _process_tick_dict(tick: SchedulerProcessTick) -> dict[str, int]:
    return {
        "launched": tick.launched,
        "finished": tick.finished,
        "recovered": tick.recovered,
        "requeued": tick.requeued,
        "launch_failures": tick.launch_failures,
    }


def cmd_delivery(args: argparse.Namespace) -> int:
    publisher = DeliveryPublisher(Path(args.output_dir))
    if args.delivery_command == "list":
        print(json.dumps(publisher.list_artifacts(), ensure_ascii=False, indent=2))
        return 0
    if args.delivery_command == "alias":
        alias = publisher.publish_source_alias(args.source_video)
        print(str(alias) if alias else "alias not created")
        return 0
    if args.delivery_command == "clean-stale":
        removed = publisher.remove_stale_subtitles()
        print(json.dumps([str(path) for path in removed], ensure_ascii=False, indent=2))
        return 0
    if args.delivery_command == "subtitles":
        generator = _subtitle_generator_from_args(args)
        rows = generator.load_rows_json(args.rows_json)
        written = generator.write_source_timeline_subtitles(rows, args.output_dir)
        print(json.dumps({role.value: str(path) for role, path in written.items()}, ensure_ascii=False, indent=2))
        return 0
    if args.delivery_command == "dub-subtitle":
        generator = _subtitle_generator_from_args(args)
        path, timeline = generator.write_dub_subtitle_from_json(args.segments_json, args.output_dir)
        print(json.dumps({"path": str(path), "segments": len(timeline.segments), "warnings": list(timeline.warnings)}, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"Unknown delivery command: {args.delivery_command}")


def cmd_media(args: argparse.Namespace) -> int:
    provider = FfmpegMediaProvider()
    if args.media_command == "probe":
        info = provider.probe(args.path)
        print(
            json.dumps(
                {
                    "path": str(info.path),
                    "duration_sec": info.duration_sec,
                    "format_name": info.format_name,
                    "bit_rate": info.bit_rate,
                    "has_video": info.has_video,
                    "has_audio": info.has_audio,
                    "video": None
                    if info.video is None
                    else {
                        "codec": info.video.codec,
                        "width": info.video.width,
                        "height": info.video.height,
                        "duration_sec": info.video.duration_sec,
                        "frame_rate": info.video.frame_rate,
                    },
                    "audio": None
                    if info.audio is None
                    else {
                        "codec": info.audio.codec,
                        "channels": info.audio.channels,
                        "sample_rate_hz": info.audio.sample_rate_hz,
                        "duration_sec": info.audio.duration_sec,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise ValueError(f"Unknown media command: {args.media_command}")


def cmd_asr(args: argparse.Namespace) -> int:
    if args.asr_command == "normalize":
        data = json.loads(Path(args.segments_json).read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("segments", [])
        segments = [
            AsrSegment(
                id=int(item.get("id", index)),
                start_sec=float(item.get("start_sec", item.get("start", 0))),
                end_sec=float(item.get("end_sec", item.get("end", 0))),
                text=str(item.get("text") or item.get("source") or ""),
                words=tuple(item.get("words") or ()),
            )
            for index, item in enumerate(data, 1)
        ]
        normalized, warnings = normalize_asr_segments(segments)
        rows = asr_segments_to_subtitle_rows(normalized)
        print(
            json.dumps(
                {
                    "segments": [segment.to_dict() for segment in normalized],
                    "subtitle_rows": [
                        {
                            "start_sec": row.start_sec,
                            "end_sec": row.end_sec,
                            "source": row.source,
                            "target": row.target,
                        }
                        for row in rows
                    ],
                    "warnings": warnings,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    raise ValueError(f"Unknown asr command: {args.asr_command}")


def cmd_llm(args: argparse.Namespace) -> int:
    config = ConfigLoader(args.config).load() if args.config else ConfigLoader().load()
    base_url = args.base_url or config.api.base_url
    model = args.model or config.api.model
    api_key = args.api_key if args.api_key is not None else config.api.key
    client = OpenAICompatibleLlmClient(
        OpenAICompatibleSettings(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_sec=float(args.timeout),
            response_format_json=config.api.llm_support_json,
            user_agent=config.api.user_agent,
            trust_env_proxy=config.api.trust_env_proxy,
        )
    )
    if args.llm_command == "models":
        print(json.dumps(client.list_models(), ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"Unknown llm command: {args.llm_command}")


def cmd_dubbing(args: argparse.Namespace) -> int:
    if args.dubbing_command == "plan":
        generator = _subtitle_generator_from_args(args)
        inputs = generator.load_timeline_inputs_json(args.segments_json)
        from eistara.core.timeline import build_dub_timeline

        timeline = build_dub_timeline(inputs)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dub_subtitle = generator.write_dub_timeline_subtitle(timeline, output_dir)
        plan = DubbingRenderService().render_plan(
            timeline,
            args.source_video,
            output_dir,
            background_audio=args.background_audio,
            dub_subtitle=dub_subtitle,
        )
        audio_mix_plan_path = output_internal_path(output_dir, "audio_mix_plan.json")
        render_plan_path = output_internal_path(output_dir, "dubbing_render_plan.json")
        audio_mix_plan_path.parent.mkdir(parents=True, exist_ok=True)
        audio_mix_plan_path.write_text(json.dumps(plan.audio_mix.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        render_plan_path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        data = plan.to_dict()
        data["audio_mix_plan_path"] = str(audio_mix_plan_path)
        data["render_plan_path"] = str(render_plan_path)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        return 0
    raise ValueError(f"Unknown dubbing command: {args.dubbing_command}")


def _subtitle_generator_from_args(args: argparse.Namespace) -> SubtitleDeliveryGenerator:
    config = ConfigLoader(args.config).load() if getattr(args, "config", None) else ConfigLoader().load()
    return SubtitleDeliveryGenerator.from_config(config.raw)


def cmd_render(args: argparse.Namespace) -> int:
    renderer = FfmpegDubbingRenderer(ffmpeg_path=args.ffmpeg)
    if args.render_command == "audio-mix":
        data = json.loads(Path(args.plan_json).read_text(encoding="utf-8-sig"))
        from eistara.core.dubbing import AudioClipPlacement, AudioMixPlan

        plan = AudioMixPlan(
            clips=tuple(
                AudioClipPlacement(
                    segment_id=str(item["segment_id"]),
                    audio_path=Path(item["audio_path"]),
                    start_sec=float(item["start_sec"]),
                    end_sec=float(item["end_sec"]),
                    gain_db=float(item.get("gain_db") or 0),
                )
                for item in data.get("clips", [])
            ),
            output_audio=Path(data["output_audio"]),
            duration_sec=float(data["duration_sec"]),
            sample_rate_hz=int(data.get("sample_rate_hz") or 44100),
            channels=int(data.get("channels") or 2),
            bitrate=str(data.get("bitrate") or "192k"),
            pre_speed_duration_sec=(
                float(data["pre_speed_duration_sec"])
                if data.get("pre_speed_duration_sec") is not None
                else None
            ),
            global_audio_speed=float(data.get("global_audio_speed") or 1.0),
            background_audio=Path(data["background_audio"]) if data.get("background_audio") else None,
            background_gain_db=float(data.get("background_gain_db") or -18.0),
            clip_lowpass_hz=int(data.get("clip_lowpass_hz") or 6800),
            clip_peak_normalize_dbfs=(
                float(data["clip_peak_normalize_dbfs"])
                if data.get("clip_peak_normalize_dbfs") is not None
                else None
            ),
            clip_fade_in_ms=int(data.get("clip_fade_in_ms") or 5),
            clip_fade_out_ms=int(data.get("clip_fade_out_ms") or 220),
            clip_tail_pad_ms=int(data.get("clip_tail_pad_ms") or 220),
            clip_tail_cleanup=bool(data.get("clip_tail_cleanup", True)),
            clip_tail_cleanup_ms=int(data.get("clip_tail_cleanup_ms") or 420),
            clip_tail_cleanup_lowpass_hz=int(data.get("clip_tail_cleanup_lowpass_hz") or 3600),
            warnings=tuple(data.get("warnings") or ()),
        )
        if args.dry_run:
            print(json.dumps({"ffmpeg_args": list(build_audio_mix_ffmpeg_args(plan, args.ffmpeg))}, ensure_ascii=False, indent=2))
            return 0
        result = renderer.render_audio_mix(plan)
        print(json.dumps(_media_result_dict(result), ensure_ascii=False, indent=2))
        return 0 if result.ok else 1
    if args.render_command == "compose":
        from eistara.core.media import build_compose_video_plan

        plan = build_compose_video_plan(args.source_video, args.dub_audio, args.output_video, args.subtitle_path)
        if args.dry_run:
            print(json.dumps({"ffmpeg_args": list(plan.ffmpeg_args(args.ffmpeg))}, ensure_ascii=False, indent=2))
            return 0
        result = renderer.render_video(plan)
        print(json.dumps(_media_result_dict(result), ensure_ascii=False, indent=2))
        return 0 if result.ok else 1
    raise ValueError(f"Unknown render command: {args.render_command}")


def _media_result_dict(result) -> dict[str, Any]:
    return {
        "command": list(result.command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ok": result.ok,
    }


def cmd_quality(args: argparse.Namespace) -> int:
    if args.quality_command == "check":
        timeline = _load_timeline(args.dub_segments_json)
        audio_mix_plan = None
        if timeline is not None:
            from eistara.core.dubbing import build_audio_mix_plan

            audio_mix_plan = build_audio_mix_plan(timeline, Path(args.output_audio or "dub.mp3"))
        report = QualityGateService(target_language=args.target_language).check(
            translations=_load_translations(args.translations_json),
            subtitle_rows=_load_subtitle_rows(args.subtitle_rows_json),
            timeline=timeline,
            audio_mix_plan=audio_mix_plan,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if report.passed else 2
    raise ValueError(f"Unknown quality command: {args.quality_command}")


def cmd_webui(args: argparse.Namespace) -> int:
    script = str(Path(__file__).resolve().parents[1] / "webui" / "main.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        script,
    ]
    if args.server_port:
        command.extend(["--server.port", str(args.server_port)])
    env = os.environ.copy()
    env["EISTARA_JOBS_DIR"] = str(args.jobs_dir)
    if args.config:
        env["EISTARA_CONFIG"] = str(args.config)
    if args.preset:
        env["EISTARA_PRESET"] = str(args.preset)
    try:
        return subprocess.call(command, env=env)
    except FileNotFoundError:
        print("Streamlit is required for the WebUI. Install streamlit or use CLI commands.")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eistara")
    parser.add_argument("--jobs-dir", default=os.fspath(Path("jobs")))
    parser.add_argument("--config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status").set_defaults(func=cmd_status)
    events = sub.add_parser("events")
    events.add_argument("--job-id")
    events.add_argument("--limit", type=int)
    events.set_defaults(func=cmd_events)
    sub.add_parser("recover").set_defaults(func=cmd_recover)
    retry = sub.add_parser("retry")
    retry.add_argument("job_id")
    retry.add_argument("--preset", choices=PIPELINE_PRESETS, default="production")
    retry.set_defaults(func=cmd_retry)
    reset = sub.add_parser("reset")
    reset.add_argument("job_id")
    reset.add_argument("--stage", choices=[stage.value for stage in StageName], required=True)
    reset.add_argument("--preset", choices=PIPELINE_PRESETS, default="production")
    reset.set_defaults(func=cmd_reset)
    health = sub.add_parser("health")
    health.add_argument("--llm-base-url")
    health.add_argument("--llm-model")
    health.add_argument("--tts-api-url")
    health.add_argument("--tts-label", default="TTS")
    health.set_defaults(func=cmd_health)
    sub.add_parser("dependency-report").set_defaults(func=cmd_dependency_report)
    youtube = sub.add_parser("youtube")
    youtube_sub = youtube.add_subparsers(dest="youtube_command", required=True)
    youtube_cookies = youtube_sub.add_parser("cookies")
    youtube_cookies_sub = youtube_cookies.add_subparsers(dest="cookies_action", required=True)
    youtube_cookies_detect = youtube_cookies_sub.add_parser("detect")
    youtube_cookies_detect.add_argument("--browser-hint")
    youtube_cookies_detect.set_defaults(func=cmd_youtube)
    youtube_cookies_apply = youtube_cookies_sub.add_parser("apply")
    youtube_cookies_apply.add_argument("--browser", default="auto")
    youtube_cookies_apply.add_argument("--profile", default="")
    youtube_cookies_apply.add_argument("--dry-run", action="store_true")
    youtube_cookies_apply.set_defaults(func=cmd_youtube)
    stages = sub.add_parser("stages")
    stages.set_defaults(func=cmd_stages)
    create = sub.add_parser("create")
    create.add_argument("input")
    create.set_defaults(func=cmd_create)
    webui = sub.add_parser("webui")
    webui.add_argument("--server-port", type=int)
    webui.add_argument("--preset", choices=PIPELINE_PRESETS, default="production")
    webui.set_defaults(func=cmd_webui)
    run_once = sub.add_parser("run-once")
    run_once.add_argument("--preset", choices=PIPELINE_PRESETS, default="production")
    run_once.add_argument("--render-audio", action="store_true")
    run_once.add_argument("--render-video", action="store_true")
    run_once.add_argument("--max-stage-retries", type=int)
    run_once.add_argument("--clear-lock", action="store_true")
    run_once.add_argument("--processes", action="store_true", help="Run selected stages in child worker processes.")
    run_once.add_argument("--in-process", action="store_true", help="Run one stage in the current process.")
    run_once.set_defaults(func=cmd_run)
    run_loop = sub.add_parser("run-loop")
    run_loop.add_argument("--preset", choices=PIPELINE_PRESETS, default="production")
    run_loop.add_argument("--render-audio", action="store_true")
    run_loop.add_argument("--render-video", action="store_true")
    run_loop.add_argument("--max-stage-retries", type=int)
    run_loop.add_argument("--clear-lock", action="store_true")
    run_loop.add_argument("--poll-interval", type=float, default=1.0)
    run_loop.add_argument("--max-iterations", type=int)
    run_loop.add_argument("--stop-when-idle", action="store_true")
    run_loop.add_argument("--processes", action="store_true", help="Run selected stages in child worker processes.")
    run_loop.add_argument("--in-process", action="store_true", help="Run stages in the current process instead of the process scheduler.")
    run_loop.set_defaults(func=cmd_run)

    delivery = sub.add_parser("delivery")
    delivery_sub = delivery.add_subparsers(dest="delivery_command", required=True)
    delivery_list = delivery_sub.add_parser("list")
    delivery_list.add_argument("output_dir")
    delivery_list.set_defaults(func=cmd_delivery)
    delivery_alias = delivery_sub.add_parser("alias")
    delivery_alias.add_argument("output_dir")
    delivery_alias.add_argument("source_video")
    delivery_alias.set_defaults(func=cmd_delivery)
    delivery_clean = delivery_sub.add_parser("clean-stale")
    delivery_clean.add_argument("output_dir")
    delivery_clean.set_defaults(func=cmd_delivery)
    delivery_subtitles = delivery_sub.add_parser("subtitles")
    delivery_subtitles.add_argument("output_dir")
    delivery_subtitles.add_argument("rows_json")
    delivery_subtitles.set_defaults(func=cmd_delivery)
    delivery_dub_subtitles = delivery_sub.add_parser("dub-subtitle")
    delivery_dub_subtitles.add_argument("output_dir")
    delivery_dub_subtitles.add_argument("segments_json")
    delivery_dub_subtitles.set_defaults(func=cmd_delivery)

    media = sub.add_parser("media")
    media_sub = media.add_subparsers(dest="media_command", required=True)
    media_probe = media_sub.add_parser("probe")
    media_probe.add_argument("path")
    media_probe.set_defaults(func=cmd_media)

    asr = sub.add_parser("asr")
    asr_sub = asr.add_subparsers(dest="asr_command", required=True)
    asr_normalize = asr_sub.add_parser("normalize")
    asr_normalize.add_argument("segments_json")
    asr_normalize.set_defaults(func=cmd_asr)

    llm = sub.add_parser("llm")
    llm_sub = llm.add_subparsers(dest="llm_command", required=True)
    llm_models = llm_sub.add_parser("models")
    llm_models.add_argument("--base-url")
    llm_models.add_argument("--model")
    llm_models.add_argument("--api-key")
    llm_models.add_argument("--timeout", default=30)
    llm_models.set_defaults(func=cmd_llm)

    dubbing = sub.add_parser("dubbing")
    dubbing_sub = dubbing.add_subparsers(dest="dubbing_command", required=True)
    dubbing_plan = dubbing_sub.add_parser("plan")
    dubbing_plan.add_argument("segments_json")
    dubbing_plan.add_argument("source_video")
    dubbing_plan.add_argument("output_dir")
    dubbing_plan.add_argument("--background-audio")
    dubbing_plan.set_defaults(func=cmd_dubbing)

    render = sub.add_parser("render")
    render_sub = render.add_subparsers(dest="render_command", required=True)
    render_audio = render_sub.add_parser("audio-mix")
    render_audio.add_argument("plan_json")
    render_audio.add_argument("--ffmpeg", default="ffmpeg")
    render_audio.add_argument("--dry-run", action="store_true")
    render_audio.set_defaults(func=cmd_render)
    render_compose = render_sub.add_parser("compose")
    render_compose.add_argument("source_video")
    render_compose.add_argument("dub_audio")
    render_compose.add_argument("output_video")
    render_compose.add_argument("--subtitle-path")
    render_compose.add_argument("--ffmpeg", default="ffmpeg")
    render_compose.add_argument("--dry-run", action="store_true")
    render_compose.set_defaults(func=cmd_render)

    quality = sub.add_parser("quality")
    quality_sub = quality.add_subparsers(dest="quality_command", required=True)
    quality_check = quality_sub.add_parser("check")
    quality_check.add_argument("--translations-json")
    quality_check.add_argument("--subtitle-rows-json")
    quality_check.add_argument("--dub-segments-json")
    quality_check.add_argument("--output-audio")
    quality_check.add_argument("--target-language", default="Simplified Chinese")
    quality_check.set_defaults(func=cmd_quality)

    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
