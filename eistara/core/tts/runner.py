from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

from eistara.core.jobs.models import StageName
from eistara.core.media import MediaProbe
from eistara.core.pipeline import StageContext, StageResult

from .models import TtsRequest, TtsSettings
from .providers import TtsProvider
from .service import TtsService
from .speed import adjust_audio_speed_in_place, v1_slow_fast_speed_factor


@dataclass(slots=True)
class TtsStageRunner:
    provider: TtsProvider
    settings: TtsSettings = TtsSettings()
    media_probe: MediaProbe | None = None
    stage: StageName = StageName.TTS

    def run(self, context: StageContext) -> StageResult:
        segments = context.task.get("tts_segments") or context.artifacts.get("tts_segments") or []
        if not segments:
            return StageResult(status="skipped", skipped=True, warnings=["No tts_segments in task or artifacts"])

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
        for index, segment in enumerate(segments, start=1):
            segment_id = segment.get("id", index)
            output_path = Path(segment.get("output_path") or context.job_dir / "output" / "audio" / "tmp" / f"{segment_id}_0_temp.wav")
            if not output_path.is_absolute():
                output_path = context.job_dir / output_path
            result = service.synthesize(
                TtsRequest(
                    text=str(segment.get("text") or ""),
                    output_path=output_path,
                    segment_id=segment_id,
                    voice=segment.get("voice"),
                    speaker=_speaker_id(segment.get("speaker", segment.get("speaker_id") or (segment.get("metadata") or {}).get("speaker"))),
                    metadata=dict(segment.get("metadata") or {}),
                )
            )
            output_path_text = str(result.output_path)
            outputs[str(segment_id)] = output_path_text
            segment["audio_path"] = output_path_text
            pre_adjust_duration = _probe_duration(self.media_probe, result.output_path)
            speed_factor = v1_slow_fast_speed_factor(str(segment.get("text") or ""), pre_adjust_duration or 0.0, settings.audio_config)
            if speed_factor < 0.999:
                adjust_audio_speed_in_place(
                    result.output_path,
                    speed_factor,
                    ffmpeg_path=str(settings.audio_config.get("ffmpeg_path") or "ffmpeg"),
                )
                warnings.append(f"{segment_id}: slowed fast TTS segment with speed factor {speed_factor:.3f}")
            duration = _probe_duration(self.media_probe, result.output_path)
            if duration is not None:
                durations[str(segment_id)] = duration
            warnings.extend(result.warnings)
            print(
                f"TTS progress: {index}/{total} segment={segment_id} cached={str(result.cached).lower()} output={result.output_path}",
                flush=True,
            )
            sys.stdout.flush()
        tts_tasks = context.task.get("tts_tasks") or context.artifacts.get("tts_tasks")
        if tts_tasks and durations:
            _write_real_durations(Path(tts_tasks), durations)
        quality_report = _write_audio_quality_report(output_dir, segments, outputs, durations)
        return StageResult(
            outputs={
                "tts_outputs": outputs,
                "tts_count": len(outputs),
                "tts_audio_quality_report": str(quality_report),
                **({"tts_durations": durations} if durations else {}),
            },
            warnings=warnings,
        )


def _probe_duration(media_probe: MediaProbe | None, path: Path) -> float | None:
    if media_probe is None:
        return None
    try:
        info = media_probe.probe(str(path))
    except Exception:
        return None
    duration = info.duration_sec or (info.audio.duration_sec if info.audio else None)
    return float(duration) if duration is not None else None


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
) -> Path:
    report_path = output_dir / "log" / "tts_audio_quality.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for segment in segments:
        segment_id = str(segment.get("id") or "")
        rows.append(
            {
                "segment_id": segment_id,
                "text": str(segment.get("text") or ""),
                "speaker": _speaker_id(segment.get("speaker", segment.get("speaker_id") or (segment.get("metadata") or {}).get("speaker"))),
                "audio_path": outputs.get(segment_id) or str(segment.get("audio_path") or ""),
                "duration_sec": durations.get(segment_id),
            }
        )
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "segment_count": len(rows),
                    "duration_count": len(durations),
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
