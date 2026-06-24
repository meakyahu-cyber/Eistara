from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from eistara.compat.enum import StrEnum


class ArtifactRole(StrEnum):
    FINAL_VIDEO = "final_video"
    DUB_SUBTITLE = "dub_subtitle"
    DUB_TARGET_SOURCE_SUBTITLE = "dub_target_source_subtitle"
    TARGET_SUBTITLE = "target_subtitle"
    SOURCE_SUBTITLE = "source_subtitle"
    TARGET_SOURCE_SUBTITLE = "target_source_subtitle"
    SOURCE_TARGET_SUBTITLE = "source_target_subtitle"
    SOURCE_ALIAS_SUBTITLE = "source_alias_subtitle"


class SubtitleColumn(StrEnum):
    SOURCE = "source"
    TARGET = "target"


@dataclass(frozen=True, slots=True)
class DeliveryArtifact:
    role: ArtifactRole
    filename: str
    kind: str
    timeline: str
    purpose: str
    source_role: ArtifactRole | None = None
    subtitle_columns: tuple[SubtitleColumn, ...] = ()

    def path(self, output_dir: str | Path) -> Path:
        return Path(output_dir) / self.filename


@dataclass(frozen=True, slots=True)
class DeliveryProfile:
    artifacts: tuple[DeliveryArtifact, ...]

    def by_role(self, role: ArtifactRole) -> DeliveryArtifact:
        for artifact in self.artifacts:
            if artifact.role == role:
                return artifact
        raise KeyError(f"Unknown delivery role: {role}")

    def known_filenames(self) -> set[str]:
        return {artifact.filename for artifact in self.artifacts}


def default_delivery_profile() -> DeliveryProfile:
    return DeliveryProfile(
        artifacts=(
            DeliveryArtifact(
                ArtifactRole.FINAL_VIDEO,
                "output_dub.mp4",
                "video",
                "dub",
                "Dubbed final video",
            ),
            DeliveryArtifact(
                ArtifactRole.DUB_SUBTITLE,
                "output_dub.srt",
                "subtitle",
                "dub",
                "Subtitle for output_dub.mp4",
                subtitle_columns=(SubtitleColumn.TARGET,),
            ),
            DeliveryArtifact(
                ArtifactRole.DUB_TARGET_SOURCE_SUBTITLE,
                "output_dub_trans_src.srt",
                "subtitle",
                "dub",
                "Target language first, source language second subtitle for output_dub.mp4",
                subtitle_columns=(SubtitleColumn.TARGET, SubtitleColumn.SOURCE),
            ),
            DeliveryArtifact(
                ArtifactRole.TARGET_SUBTITLE,
                "trans.srt",
                "subtitle",
                "source",
                "Target-language subtitle for the source video",
                subtitle_columns=(SubtitleColumn.TARGET,),
            ),
            DeliveryArtifact(
                ArtifactRole.SOURCE_SUBTITLE,
                "src.srt",
                "subtitle",
                "source",
                "Source-language subtitle for the source video",
                subtitle_columns=(SubtitleColumn.SOURCE,),
            ),
            DeliveryArtifact(
                ArtifactRole.TARGET_SOURCE_SUBTITLE,
                "trans_src.srt",
                "subtitle",
                "source",
                "Target language first, source language second",
                subtitle_columns=(SubtitleColumn.TARGET, SubtitleColumn.SOURCE),
            ),
            DeliveryArtifact(
                ArtifactRole.SOURCE_TARGET_SUBTITLE,
                "src_trans.srt",
                "subtitle",
                "source",
                "Source language first, target language second",
                subtitle_columns=(SubtitleColumn.SOURCE, SubtitleColumn.TARGET),
            ),
        )
    )
