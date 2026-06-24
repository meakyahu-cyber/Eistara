from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eistara.core.jobs.models import StageName
from eistara.core.pipeline import StageContext, StageResult

from .models import AsrRequest, AsrSettings
from .providers import AsrProvider
from .service import AsrService, asr_segments_to_subtitle_rows


@dataclass(slots=True)
class AsrStageRunner:
    provider: AsrProvider
    settings: AsrSettings = AsrSettings()
    stage: StageName = StageName.TRANSCRIBE

    def run(self, context: StageContext) -> StageResult:
        audio_path = context.task.get("audio_path") or context.task.get("source_audio")
        if not audio_path:
            return StageResult(status="skipped", skipped=True, warnings=["No audio_path in task"])
        language = context.task.get("source_language") or self.settings.language
        result = AsrService(self.provider, self.settings).transcribe(
            AsrRequest(
                audio_path=Path(audio_path),
                language=str(language) if language else None,
                prompt=str(context.task.get("asr_prompt") or ""),
            )
        )
        rows = asr_segments_to_subtitle_rows(result.segments)
        return StageResult(
            outputs={
                "language": result.language,
                "segments": [segment.to_dict() for segment in result.segments],
                "subtitle_rows": [
                    {
                        "start_sec": row.start_sec,
                        "end_sec": row.end_sec,
                        "source": row.source,
                        "target": row.target,
                    }
                    for row in rows
                ],
            },
            warnings=list(result.warnings),
        )
