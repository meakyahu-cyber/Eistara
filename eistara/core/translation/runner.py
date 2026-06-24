from __future__ import annotations

from dataclasses import dataclass

from eistara.core.jobs.models import StageName
from eistara.core.pipeline import StageContext, StageResult

from .llm import LlmClient
from .models import Terminology, TranslationItem, TranslationSettings
from .service import PublishTranslationService


@dataclass(slots=True)
class TranslationStageRunner:
    llm: LlmClient
    settings: TranslationSettings = TranslationSettings()
    stage: StageName = StageName.TRANSLATE

    def run(self, context: StageContext) -> StageResult:
        raw_items = context.task.get("translation_items") or []
        if not raw_items:
            return StageResult(status="skipped", skipped=True, warnings=["No translation_items in task"])
        items = [
            TranslationItem(
                id=int(item["id"]),
                source=str(item["source"]),
                start=str(item.get("start") or ""),
                end=str(item.get("end") or ""),
                duration_sec=item.get("duration_sec"),
            )
            for item in raw_items
        ]
        terminology = Terminology(
            theme=str((context.task.get("terminology") or {}).get("theme") or ""),
            terms=tuple((context.task.get("terminology") or {}).get("terms") or ()),
        )
        result = PublishTranslationService(self.llm, self.settings).translate(items, terminology)
        return StageResult(
            outputs={
                "translations": result.translations,
                "translation_count": len(result.translations),
            },
            warnings=result.warnings,
        )
