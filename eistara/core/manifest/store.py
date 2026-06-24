from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from eistara.core.jobs.models import STAGE_ORDER, StageName, utc_now_iso

from .models import Manifest, StageRecord


MANIFEST_FILE = "manifest.json"


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, path)


class JsonManifestStore:
    def load_or_create(self, job_dir: Path, task: dict[str, Any]) -> Manifest:
        path = job_dir / MANIFEST_FILE
        data: dict[str, Any] | None = None
        should_save = False
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            data = None

        if not isinstance(data, dict):
            should_save = True
            manifest = Manifest(
                task_id=str(task.get("id") or job_dir.name),
                workdir=str(job_dir.resolve()),
                stage_order=list(STAGE_ORDER),
                input={
                    "source": task.get("source"),
                    "title": task.get("title"),
                    "source_language": task.get("source_language"),
                    "target_language": task.get("target_language"),
                },
            )
        else:
            stage_order = [StageName(str(item)) for item in data.get("stage_order") or [s.value for s in STAGE_ORDER]]
            manifest = Manifest(
                task_id=str(data.get("task_id") or task.get("id") or job_dir.name),
                workdir=str(data.get("workdir") or job_dir.resolve()),
                stage_order=stage_order,
                schema_version=int(data.get("schema_version") or 1),
                app=str(data.get("app") or "Eistara"),
                created_at=str(data.get("created_at") or utc_now_iso()),
                updated_at=str(data.get("updated_at") or utc_now_iso()),
                input=dict(data.get("input") or {}),
                outputs=dict(data.get("outputs") or {}),
                speakers=_speakers_from_manifest(data.get("speakers")),
                warnings=list(data.get("warnings") or []),
                caption_source=data.get("caption_source"),
            )
            for stage in stage_order:
                raw = (data.get("stages") or {}).get(stage.value) or {}
                manifest.stages[stage] = StageRecord(
                    name=stage,
                    status=str(raw.get("status") or "pending"),
                    attempts=int(raw.get("attempts") or 0),
                    started_at=raw.get("started_at"),
                    finished_at=raw.get("finished_at"),
                    duration_sec=raw.get("duration_sec"),
                    outputs=dict(raw.get("outputs") or {}),
                    report=raw.get("report"),
                    log=raw.get("log"),
                    error=raw.get("error"),
                )

        for stage in manifest.stage_order:
            if stage not in manifest.stages:
                manifest.stages[stage] = StageRecord(name=stage)
                should_save = True
        if should_save:
            self.save(job_dir, manifest)
        return manifest

    def save(self, job_dir: Path, manifest: Manifest) -> None:
        manifest.updated_at = utc_now_iso()
        _write_json_atomic(job_dir / MANIFEST_FILE, manifest.to_dict())

    def mark_running(self, job_dir: Path, task: dict[str, Any], stage: StageName, attempt: int, log_path: Path | None) -> None:
        manifest = self.load_or_create(job_dir, task)
        record = manifest.stages.setdefault(stage, StageRecord(name=stage))
        record.status = "running"
        record.attempts = attempt
        record.started_at = utc_now_iso()
        record.finished_at = None
        record.error = None
        record.log = str(log_path) if log_path else None
        self.save(job_dir, manifest)

    def mark_finished(
        self,
        job_dir: Path,
        task: dict[str, Any],
        stage: StageName,
        status: str,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        log_path: Path | None = None,
    ) -> None:
        manifest = self.load_or_create(job_dir, task)
        record = manifest.stages.setdefault(stage, StageRecord(name=stage))
        record.status = status
        record.finished_at = utc_now_iso()
        record.outputs = dict(outputs or {})
        record.error = error
        if log_path:
            record.log = str(log_path)
        if outputs:
            manifest.outputs.update(outputs)
            _merge_output_speakers(manifest, outputs)
        self.save(job_dir, manifest)


def _speakers_from_manifest(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list) and value:
        speakers = []
        for item in value:
            if not isinstance(item, dict):
                continue
            speaker_id = str(item.get("id") or item.get("speaker") or "").strip()
            if speaker_id:
                speakers.append(dict(item) | {"id": speaker_id})
        if speakers:
            return speakers
    return [{"id": "SPEAKER_00", "label": "Default speaker", "role": "default"}]


def _merge_output_speakers(manifest: Manifest, outputs: dict[str, Any]) -> None:
    seen = {str(item.get("id") or "") for item in manifest.speakers if isinstance(item, dict)}
    for speaker_id in _speaker_ids_from_outputs(outputs):
        if speaker_id not in seen:
            manifest.speakers.append({"id": speaker_id, "label": speaker_id, "role": "detected"})
            seen.add(speaker_id)


def _speaker_ids_from_outputs(outputs: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("segments", "subtitle_rows", "tts_segments"):
        raw_items = outputs.get(key)
        if not isinstance(raw_items, list):
            continue
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            speaker = str(item.get("speaker") or item.get("speaker_id") or "").strip()
            if speaker and speaker not in values:
                values.append(speaker)
    return values
