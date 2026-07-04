from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from eistara.core.jobs.models import StageName

INTERNAL_OUTPUT_DIRNAME = "internal"


def output_internal_dir(output_dir: Path) -> Path:
    return Path(output_dir) / INTERNAL_OUTPUT_DIRNAME


def output_internal_path(output_dir: Path, filename: str) -> Path:
    return output_internal_dir(output_dir) / filename


def resolve_task_output_dir(job_dir: str | Path, task: Mapping[str, Any] | None = None) -> Path:
    job_root = Path(job_dir)
    output_dir = Path((task or {}).get("output_dir") or job_root / "output")
    if not output_dir.is_absolute():
        output_dir = job_root / output_dir
    return output_dir


def resolve_output_dir(context: Any) -> Path:
    return resolve_task_output_dir(context.job_dir, context.task)


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    name: str
    relative_path: str
    required: bool = True

    def resolve(self, job_dir: Path) -> Path:
        return job_dir / self.relative_path


@dataclass(frozen=True, slots=True)
class StageArtifactContract:
    stage: StageName
    outputs: tuple[ArtifactSpec, ...]

    def expected_paths(self, job_dir: Path) -> dict[str, Path]:
        return {artifact.name: artifact.resolve(job_dir) for artifact in self.outputs}

    def existing_outputs(self, job_dir: Path) -> dict[str, str]:
        return {
            name: str(path)
            for name, path in self.expected_paths(job_dir).items()
            if path.exists()
        }

    def missing_required(self, job_dir: Path) -> list[ArtifactSpec]:
        return [
            artifact
            for artifact in self.outputs
            if artifact.required and not artifact.resolve(job_dir).exists()
        ]

    def is_satisfied(self, job_dir: Path) -> bool:
        return not self.missing_required(job_dir)


DEFAULT_ARTIFACT_CONTRACTS: dict[StageName, StageArtifactContract] = {
    StageName.DOWNLOAD: StageArtifactContract(
        StageName.DOWNLOAD,
        (
            ArtifactSpec("source_video", "output/source_video.mp4", required=False),
        ),
    ),
    StageName.TRANSCRIBE: StageArtifactContract(
        StageName.TRANSCRIBE,
        (
            ArtifactSpec("cleaned_chunks", "output/log/cleaned_chunks.xlsx"),
            ArtifactSpec("split_by_nlp", "output/log/split_by_nlp.txt", required=False),
            ArtifactSpec("subtitle_rows_json", "output/internal/subtitle_rows.json", required=False),
            ArtifactSpec("raw_audio", "output/audio/raw.mp3"),
            ArtifactSpec("high_quality_audio", "output/audio/raw_hq.wav"),
            ArtifactSpec("vocal_audio", "output/audio/vocal.mp3", required=False),
            ArtifactSpec("background_audio", "output/audio/background.mp3", required=False),
        ),
    ),
    StageName.TRANSLATE: StageArtifactContract(
        StageName.TRANSLATE,
        (
            ArtifactSpec("translation", "output/log/publish_translation.xlsx"),
            ArtifactSpec("subtitles", "output/log/publish_subtitles.xlsx"),
            ArtifactSpec("audio_script", "output/log/publish_audio_script.xlsx"),
            ArtifactSpec("publish_source_lines", "output/log/publish_source_lines.txt"),
            ArtifactSpec("source_srt", "output/src.srt"),
            ArtifactSpec("translated_srt", "output/trans.srt"),
            ArtifactSpec("audio_source_srt", "output/audio/src_subs_for_audio.srt"),
            ArtifactSpec("audio_translated_srt", "output/audio/trans_subs_for_audio.srt"),
            ArtifactSpec("translations_json", "output/internal/translations.json", required=False),
        ),
    ),
    StageName.TTS_PREPARE: StageArtifactContract(
        StageName.TTS_PREPARE,
        (
            ArtifactSpec("tts_tasks", "output/audio/tts_tasks.xlsx"),
            ArtifactSpec("reference_audio_dir", "output/audio/refers", required=False),
        ),
    ),
    StageName.TTS: StageArtifactContract(
        StageName.TTS,
        (
            ArtifactSpec("tts_tmp_dir", "output/audio/tmp"),
            ArtifactSpec("tts_audio_quality_report", "output/log/tts_audio_quality.json", required=False),
        ),
    ),
    StageName.AUDIO_MIX: StageArtifactContract(
        StageName.AUDIO_MIX,
        (
            ArtifactSpec("audio_mix_plan", "output/internal/audio_mix_plan.json", required=False),
            ArtifactSpec("dub_segments_json", "output/internal/dub_segments.json", required=False),
            ArtifactSpec("dub_audio", "output/dub.mp3"),
            ArtifactSpec("dub_subtitles", "output/output_dub.srt"),
        ),
    ),
    StageName.QUALITY: StageArtifactContract(
        StageName.QUALITY,
        (
            ArtifactSpec("quality_report", "output/quality_report.json"),
        ),
    ),
    StageName.COMPOSE: StageArtifactContract(
        StageName.COMPOSE,
        (
            ArtifactSpec("compose_plan", "output/internal/compose_plan.json", required=False),
            ArtifactSpec("dub_video", "output/output_dub.mp4"),
        ),
    ),
}


def contracts_for(stages: Iterable[StageName]) -> dict[StageName, StageArtifactContract]:
    return {stage: DEFAULT_ARTIFACT_CONTRACTS[stage] for stage in stages}
