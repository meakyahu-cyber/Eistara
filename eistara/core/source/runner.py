from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eistara.core.jobs import StageName
from eistara.core.pipeline import StageContext, StageResult

from .models import SourceRequest, SourceSettings, allowed_video_formats
from .providers import LocalFileSourceProvider, SourceProvider


GENERATED_VIDEO_NAMES = {
    "output.mp4",
    "output_dub.mp4",
    "output_sub.mp4",
    "output_dub.webm",
    "output_sub.webm",
}


@dataclass(slots=True)
class SourceStageRunner:
    url_provider: SourceProvider | None = None
    file_provider: SourceProvider = LocalFileSourceProvider()
    settings: SourceSettings = SourceSettings()
    stage: StageName = StageName.DOWNLOAD

    def run(self, context: StageContext) -> StageResult:
        output_dir = Path(context.task.get("output_dir") or context.job_dir / "output")
        existing = _existing_source_video(output_dir, self.settings)
        if existing:
            source = context.task.get("source") or context.task.get("source_video")
            return StageResult(
                outputs={
                    "source_video": str(existing),
                    "source_type": _source_type(context, source),
                },
                skipped=True,
            )
        source = context.task.get("source") or context.task.get("source_video")
        if not source:
            return StageResult(status="skipped", skipped=True, warnings=["No source in task"])
        source_type = _source_type(context, source)
        provider = self.url_provider if source_type == "url" else self.file_provider
        if provider is None:
            return StageResult(status="skipped", skipped=True, warnings=[f"No source provider for source_type={source_type}"])
        result = provider.acquire(
            SourceRequest(
                source=str(source),
                source_type=source_type,
                output_dir=output_dir,
                title=str(context.task.get("title") or ""),
                resolution=str(
                    context.task.get("resolution")
                    or self.settings.provider_config.get("resolution")
                    or self.settings.provider_config.get("ytb_resolution")
                    or ""
                ),
            ),
            self.settings,
        )
        return StageResult(
            outputs={
                "source_video": str(result.source_video),
                "source_type": result.source_type,
                **result.metadata,
            },
            warnings=list(result.warnings),
        )


def _existing_source_video(output_dir: Path, settings: SourceSettings) -> Path | None:
    if not output_dir.is_dir():
        return None
    allowed = allowed_video_formats(settings)
    candidates = [
        path
        for path in sorted(output_dir.iterdir())
        if path.is_file()
        and path.suffix.lower().lstrip(".") in allowed
        and path.name not in GENERATED_VIDEO_NAMES
        and not path.name.startswith("output_")
    ]
    return candidates[0] if len(candidates) == 1 else None


def _source_type(context: StageContext, source: object) -> str:
    return str(
        context.task.get("source_type")
        or ("url" if str(source).startswith(("http://", "https://")) else "file")
    ).lower()
