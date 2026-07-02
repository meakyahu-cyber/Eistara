from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd

from eistara.core.jobs.models import StageName
from eistara.core.media import MediaProbe, source_duration_sec
from eistara.core.pipeline import StageContext, StageResult

from .models import TtsRequest, TtsSettings
from .providers import TtsProvider
from .service import TtsService
from .audio import wav_duration_sec
from .cache import cache_meta_path
from .pacing_quality import analyze_segment_pacing, build_pacing_quality_plan, retry_improved_pacing
from .segments import load_tts_segments


@dataclass(slots=True)
class TtsStageRunner:
    provider: TtsProvider
    settings: TtsSettings = TtsSettings()
    media_probe: MediaProbe | None = None
    stage: StageName = StageName.TTS

    def run(self, context: StageContext) -> StageResult:
        segments = load_tts_segments(context)
        if not segments:
            return StageResult(status="skipped", skipped=True, warnings=["No tts_segments or tts_segments_json in task or artifacts"])

        output_dir = _resolve_output_dir(context)
        settings = self.settings
        prepare_settings = getattr(self.provider, "prepare_settings", None)
        if prepare_settings:
            settings = prepare_settings(
                settings,
                output_dir=output_dir,
                reference_audio_dir=context.task.get("reference_audio_dir") or context.artifacts.get("reference_audio_dir"),
            )

        outputs: dict[str, str] = {}
        durations: dict[str, float] = {}
        warnings: list[str] = []
        service = TtsService(self.provider, settings)
        total = len(segments)
        provider_metadata_hook = getattr(self.provider, "prepare_request_metadata", None)
        provider_retry_plan = getattr(self.provider, "build_retry_plan", None)
        provider_retry_decisions: dict[str, dict[str, Any]] = {}
        pacing_retry_decisions: dict[str, dict[str, Any]] = {}
        pacing_summary: dict[str, Any] = {"enabled": False}
        for index, segment in enumerate(segments, start=1):
            segment_id = segment.get("id", index)
            output_path = Path(segment.get("output_path") or context.job_dir / "output" / "audio" / "tmp" / f"{segment_id}_0_temp.wav")
            if not output_path.is_absolute():
                output_path = context.job_dir / output_path
            metadata = dict(segment.get("metadata") or {})
            if callable(provider_metadata_hook):
                metadata = provider_metadata_hook(segment, metadata, settings)
            result = service.synthesize(
                TtsRequest(
                    text=str(segment.get("text") or ""),
                    output_path=output_path,
                    segment_id=segment_id,
                    voice=segment.get("voice"),
                    speaker=_speaker_id(segment.get("speaker", segment.get("speaker_id") or (segment.get("metadata") or {}).get("speaker"))),
                    metadata=metadata,
                )
            )
            output_path_text = str(result.output_path)
            outputs[str(segment_id)] = output_path_text
            segment["audio_path"] = output_path_text
            duration = _probe_duration(self.media_probe, result.output_path)
            if duration is not None:
                durations[str(segment_id)] = duration
            warnings.extend(result.warnings)
            print(
                f"TTS progress: {index}/{total} segment={segment_id} cached={str(result.cached).lower()} output={result.output_path}",
                flush=True,
            )
            sys.stdout.flush()
        pacing_plan = build_pacing_quality_plan(segments, outputs, settings.audio_config)
        pacing_summary = pacing_plan.summary()
        if pacing_plan.retry_decisions:
            for retry_index, (segment_id_text, decision) in enumerate(pacing_plan.retry_decisions.items(), start=1):
                segment = next((item for item in segments if str(item.get("id") or "") == segment_id_text), None)
                if segment is None:
                    continue
                output_path = Path(segment.get("audio_path") or segment.get("output_path") or context.job_dir / "output" / "audio" / "tmp" / f"{segment_id_text}_0_temp.wav")
                if not output_path.is_absolute():
                    output_path = context.job_dir / output_path
                backup_path = output_path.with_suffix(output_path.suffix + ".pacing_retry_original")
                backup_cache_path = cache_meta_path(backup_path)
                original_cache_path = cache_meta_path(output_path)
                try:
                    shutil.copy2(output_path, backup_path)
                    if original_cache_path.exists():
                        shutil.copy2(original_cache_path, backup_cache_path)
                except OSError as exc:
                    warnings.append(f"{segment_id_text}: pacing quality retry skipped; could not backup audio: {exc}")
                    continue
                metadata = dict(segment.get("metadata") or {})
                if callable(provider_metadata_hook):
                    metadata = provider_metadata_hook(segment, metadata, settings)
                metadata["eistara_pacing_quality_retry"] = {
                    "attempt": 1,
                    "reason": decision.reason,
                    "baseline_units_per_sec": round(decision.baseline_units_per_sec, 3),
                }
                service.cache.remove(output_path)
                retry_detail = decision.to_dict()
                try:
                    result = service.synthesize(
                        TtsRequest(
                            text=str(segment.get("text") or ""),
                            output_path=output_path,
                            segment_id=segment.get("id", segment_id_text),
                            voice=segment.get("voice"),
                            speaker=_speaker_id(segment.get("speaker", segment.get("speaker_id") or (segment.get("metadata") or {}).get("speaker"))),
                            metadata=metadata,
                        )
                    )
                    retry_stats = analyze_segment_pacing(segment_id_text, str(segment.get("text") or ""), output_path, settings.audio_config)
                    keep_retry = retry_improved_pacing(decision.original, retry_stats, decision, settings.audio_config)
                    if not keep_retry:
                        shutil.copy2(backup_path, output_path)
                        if backup_cache_path.exists():
                            shutil.copy2(backup_cache_path, original_cache_path)
                        elif original_cache_path.exists():
                            original_cache_path.unlink()
                        duration = durations.get(segment_id_text)
                    else:
                        duration = _probe_duration(self.media_probe, output_path, result.duration_sec)
                        if duration is not None:
                            durations[segment_id_text] = duration
                        outputs[segment_id_text] = str(output_path)
                        segment["audio_path"] = str(output_path)
                    retry_detail["retry"] = {
                        "kept": bool(keep_retry),
                        "cached": bool(result.cached),
                        "duration_sec": duration,
                        "stats": retry_stats.to_dict(),
                    }
                    pacing_retry_decisions[segment_id_text] = retry_detail
                    warnings.extend(result.warnings)
                    print(
                        "TTS pacing retry: "
                        f"{retry_index}/{len(pacing_plan.retry_decisions)} segment={segment_id_text} "
                        f"reason={decision.reason} kept={str(keep_retry).lower()} output={output_path}",
                        flush=True,
                    )
                    sys.stdout.flush()
                except Exception as exc:
                    try:
                        shutil.copy2(backup_path, output_path)
                        if backup_cache_path.exists():
                            shutil.copy2(backup_cache_path, original_cache_path)
                    except OSError:
                        pass
                    retry_detail["retry"] = {"kept": False, "error": str(exc)}
                    pacing_retry_decisions[segment_id_text] = retry_detail
                    warnings.append(f"{segment_id_text}: pacing quality retry failed; restored original audio: {exc}")
                for stale_path in (backup_path, backup_cache_path):
                    try:
                        stale_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
        if callable(provider_retry_plan) and durations:
            retry_plan = provider_retry_plan(
                segments,
                durations,
                settings,
                source_duration_sec=source_duration_sec(context, self.media_probe),
            )
            for retry_index, (segment_id_text, decision) in enumerate(retry_plan.items(), start=1):
                segment = decision["segment"]
                output_path = Path(segment.get("audio_path") or segment.get("output_path") or context.job_dir / "output" / "audio" / "tmp" / f"{segment_id_text}_0_temp.wav")
                if not output_path.is_absolute():
                    output_path = context.job_dir / output_path
                original_duration = durations.get(segment_id_text)
                min_accept_duration = _optional_float(decision.get("min_accept_duration_sec"))
                backup_path: Path | None = None
                backup_cache_path: Path | None = None
                original_cache_path = cache_meta_path(output_path)
                if min_accept_duration is not None:
                    backup_path = output_path.with_suffix(output_path.suffix + ".provider_retry_original")
                    backup_cache_path = cache_meta_path(backup_path)
                    try:
                        shutil.copy2(output_path, backup_path)
                        if original_cache_path.exists():
                            shutil.copy2(original_cache_path, backup_cache_path)
                    except OSError as exc:
                        warnings.append(f"{segment_id_text}: provider retry rollback disabled; could not backup audio: {exc}")
                        backup_path = None
                        backup_cache_path = None
                metadata = dict(decision.get("metadata") or segment.get("metadata") or {})
                result = service.synthesize(
                    TtsRequest(
                        text=str(segment.get("text") or ""),
                        output_path=output_path,
                        segment_id=segment.get("id", segment_id_text),
                        voice=segment.get("voice"),
                        speaker=_speaker_id(segment.get("speaker", segment.get("speaker_id") or (segment.get("metadata") or {}).get("speaker"))),
                        metadata=metadata,
                    )
                )
                duration = _probe_duration(self.media_probe, result.output_path)
                keep_retry = True
                if min_accept_duration is not None and duration is not None and duration < min_accept_duration:
                    keep_retry = False
                    if backup_path is None:
                        keep_retry = True
                    else:
                        try:
                            shutil.copy2(backup_path, output_path)
                            if backup_cache_path is not None and backup_cache_path.exists():
                                shutil.copy2(backup_cache_path, original_cache_path)
                            elif original_cache_path.exists():
                                original_cache_path.unlink()
                        except OSError as exc:
                            warnings.append(f"{segment_id_text}: provider retry restore failed after short result: {exc}")
                            keep_retry = True
                    if not keep_retry:
                        duration = original_duration
                if duration is not None:
                    durations[segment_id_text] = duration
                outputs[segment_id_text] = str(result.output_path)
                segment["audio_path"] = str(result.output_path)
                decision = dict(decision)
                decision.pop("segment", None)
                decision.pop("metadata", None)
                decision["retry_duration_sec"] = duration
                decision["cached"] = bool(result.cached)
                if min_accept_duration is not None:
                    decision["kept"] = bool(keep_retry)
                    if not keep_retry:
                        decision["rejected_reason"] = "retry_duration_below_min_accept"
                provider_retry_decisions[segment_id_text] = decision
                warnings.extend(result.warnings)
                target_label = f" target={float(decision['target_duration_sec']):.3f}s" if "target_duration_sec" in decision else ""
                print(
                    "TTS provider retry: "
                    f"{retry_index}/{len(retry_plan)} segment={segment_id_text} "
                    f"{target_label} cached={str(result.cached).lower()} output={result.output_path}",
                    flush=True,
                )
                sys.stdout.flush()
                for stale_path in (backup_path, backup_cache_path):
                    if stale_path is None:
                        continue
                    try:
                        stale_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        pass
        tts_tasks = context.task.get("tts_tasks") or context.artifacts.get("tts_tasks")
        if tts_tasks and durations:
            _write_real_durations(Path(tts_tasks), durations)
        quality_report = _write_audio_quality_report(
            output_dir,
            segments,
            outputs,
            durations,
            provider_retry_decisions,
            pacing_summary,
            pacing_retry_decisions,
        )
        return StageResult(
            outputs={
                "tts_outputs": outputs,
                "tts_count": len(outputs),
                "tts_audio_quality_report": str(quality_report),
                "tts_provider_retry_count": len(provider_retry_decisions),
                "tts_pacing_retry_count": len(pacing_retry_decisions),
                **({"tts_durations": durations} if durations else {}),
            },
            warnings=warnings,
        )


def _probe_duration(media_probe: MediaProbe | None, path: Path, fallback: float | None = None) -> float | None:
    if media_probe is None:
        duration = fallback if fallback is not None else wav_duration_sec(path)
        return float(duration) if duration is not None else None
    try:
        info = media_probe.probe(str(path))
        duration = info.duration_sec or (info.audio.duration_sec if info.audio else None)
        if duration is not None:
            return float(duration)
    except Exception:
        pass
    duration = fallback if fallback is not None else wav_duration_sec(path)
    return float(duration) if duration is not None else None


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _write_real_durations(tts_tasks: Path, durations: dict[str, float]) -> None:
    if not tts_tasks.exists():
        return
    df = pd.read_excel(tts_tasks)
    if "real_dur" not in df.columns:
        df["real_dur"] = 0.0
    for index, row in df.iterrows():
        key = str(row.get("number"))
        if key.endswith(".0"):
            key = key[:-2]
        if key in durations:
            df.at[index, "real_dur"] = durations[key]
            continue
        prefix = f"{key}_"
        row_duration = sum(float(duration) for segment_id, duration in durations.items() if str(segment_id).startswith(prefix))
        if row_duration > 0:
            df.at[index, "real_dur"] = row_duration
    df.to_excel(tts_tasks, index=False)


def _write_audio_quality_report(
    output_dir: Path,
    segments: list[dict],
    outputs: dict[str, str],
    durations: dict[str, float],
    provider_retry_decisions: dict[str, dict[str, Any]] | None = None,
    pacing_summary: dict[str, Any] | None = None,
    pacing_retry_decisions: dict[str, dict[str, Any]] | None = None,
) -> Path:
    report_path = output_dir / "log" / "tts_audio_quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    provider_retry_decisions = provider_retry_decisions or {}
    pacing_retry_decisions = pacing_retry_decisions or {}
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        rows.append(
            {
                "segment_id": segment_id,
                "text": str(segment.get("text") or ""),
                "speaker": _speaker_id(segment.get("speaker", segment.get("speaker_id") or (segment.get("metadata") or {}).get("speaker"))),
                "audio_path": outputs.get(segment_id) or str(segment.get("audio_path") or ""),
                "duration_sec": durations.get(segment_id),
                "provider_retry": provider_retry_decisions.get(segment_id),
                "pacing_retry": pacing_retry_decisions.get(segment_id),
            }
        )
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "segment_count": len(rows),
                    "duration_count": len(durations),
                    "provider_retry_count": len(provider_retry_decisions),
                    "pacing_retry_count": len(pacing_retry_decisions),
                    "pacing": pacing_summary or {"enabled": False},
                },
                "segments": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report_path


def _resolve_output_dir(context: StageContext) -> Path:
    output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
    if not output_dir.is_absolute():
        output_dir = context.job_dir / output_dir
    return output_dir


def _speaker_id(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "SPEAKER_00"
