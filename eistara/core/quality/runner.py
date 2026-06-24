from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eistara.core.delivery import SubtitleDeliveryGenerator, SubtitleRow
from eistara.core.dubbing import build_audio_mix_plan
from eistara.core.jobs.models import StageName
from eistara.core.pipeline import StageContext, StageResult
from eistara.core.timeline import build_dub_timeline

from .service import QualityGateService


@dataclass(slots=True)
class QualityStageRunner:
    service: QualityGateService = QualityGateService()
    stage: StageName = StageName.QUALITY

    def run(self, context: StageContext) -> StageResult:
        translations_json = context.task.get("translations_json") or context.artifacts.get("translations_json")
        subtitle_rows_json = context.task.get("subtitle_rows_json") or context.artifacts.get("subtitle_rows_json")
        dub_segments_json = context.task.get("dub_segments_json") or context.artifacts.get("dub_segments_json")
        report = self.service.check(
            translations=_load_translations(translations_json),
            subtitle_rows=_load_subtitle_rows(subtitle_rows_json),
            timeline=_load_timeline(dub_segments_json),
            audio_mix_plan=(
                build_audio_mix_plan(_load_timeline(dub_segments_json), context.job_dir / "output" / "dub.mp3")
                if dub_segments_json
                else None
            ),
        )
        output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "quality_report.json"
        report_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return StageResult(
            status="done" if report.passed else "failed",
            outputs={"quality_report": str(report_path), **report.to_dict()},
            warnings=[issue.message for issue in report.issues if issue.severity.value == "warning"],
        )


def _load_translations(path: str | Path | None) -> dict[int, str] | None:
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "translations" in data:
        data = data["translations"]
    if isinstance(data, dict):
        return {int(key): str(value.get("text") if isinstance(value, dict) else value) for key, value in data.items()}
    if isinstance(data, list):
        return {int(item["id"]): str(item.get("text") or "") for item in data}
    raise ValueError("translations_json must be a dict, list, or object with translations")


def _load_subtitle_rows(path: str | Path | None) -> list[SubtitleRow] | None:
    if not path:
        return None
    return SubtitleDeliveryGenerator().load_rows_json(path)


def _load_timeline(path: str | Path | None):
    if not path:
        return None
    inputs = SubtitleDeliveryGenerator().load_timeline_inputs_json(path)
    return build_dub_timeline(inputs)
