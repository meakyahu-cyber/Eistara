from __future__ import annotations

import json
from pathlib import Path

from eistara.core.jobs import StageName
from eistara.core.manifest import JsonManifestStore


def test_manifest_load_existing_does_not_rewrite_file(tmp_path: Path) -> None:
    store = JsonManifestStore()
    task = {"id": "job_0001", "source": "source.mp4", "title": "Demo"}
    store.load_or_create(tmp_path, task)
    manifest_path = tmp_path / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["updated_at"] = "2000-01-01T00:00:00+00:00"
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    before = manifest_path.read_text(encoding="utf-8")

    manifest = store.load_or_create(tmp_path, task)

    assert manifest.updated_at == "2000-01-01T00:00:00+00:00"
    assert manifest_path.read_text(encoding="utf-8") == before


def test_manifest_tracks_speakers_from_stage_outputs(tmp_path: Path) -> None:
    store = JsonManifestStore()
    task = {"id": "job_0001", "source": "source.mp4", "title": "Demo"}

    store.mark_finished(
        tmp_path,
        task,
        StageName.TRANSCRIBE,
        "done",
        outputs={"subtitle_rows": [{"source": "hello", "speaker": "SPEAKER_01"}]},
    )

    manifest = store.load_or_create(tmp_path, task)
    speaker_ids = [item["id"] for item in manifest.speakers]

    assert "SPEAKER_00" in speaker_ids
    assert "SPEAKER_01" in speaker_ids
