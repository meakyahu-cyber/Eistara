from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .profile import ArtifactRole, DeliveryArtifact, DeliveryProfile, default_delivery_profile


STALE_SUBTITLE_NAMES = {
    "source_bilingual.srt",
    "source_zh.srt",
    "source_en.srt",
    "en.srt",
    "zh.srt",
    "en_zh.srt",
    "zh_en.srt",
    "dub.srt",
}


@dataclass(slots=True)
class DeliveryPublisher:
    output_dir: Path
    profile: DeliveryProfile = field(default_factory=default_delivery_profile)

    def list_artifacts(self) -> list[dict[str, object]]:
        artifacts = [self._artifact_info(artifact, artifact.path(self.output_dir)) for artifact in self.profile.artifacts]
        artifacts.extend(self._source_alias_infos())
        return artifacts

    def remove_stale_subtitles(self) -> list[Path]:
        removed: list[Path] = []
        for name in STALE_SUBTITLE_NAMES:
            path = self.output_dir / name
            try:
                path.unlink()
                removed.append(path)
            except FileNotFoundError:
                pass
        return removed

    def publish_source_alias(self, source_video: str | Path) -> Path | None:
        source_video_path = Path(source_video)
        alias_path = source_video_path.with_suffix(".srt")
        if not alias_path.is_absolute():
            alias_path = self.output_dir / alias_path.name
        source_artifact = self.profile.by_role(ArtifactRole.TARGET_SOURCE_SUBTITLE).path(self.output_dir)
        if not source_artifact.exists():
            return None
        alias_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_artifact, alias_path)
        return alias_path

    def _source_alias_infos(self) -> list[dict[str, object]]:
        known = self.profile.known_filenames()
        aliases = sorted(path for path in self.output_dir.glob("*.srt") if path.name not in known)
        return [
            self._artifact_info(
                DeliveryArtifact(
                    ArtifactRole.SOURCE_ALIAS_SUBTITLE,
                    path.name,
                    "subtitle",
                    "source",
                    "Convenience copy of trans_src.srt named after the source video",
                    source_role=ArtifactRole.TARGET_SOURCE_SUBTITLE,
                ),
                path,
            )
            for path in aliases
        ]

    def _artifact_info(self, artifact: DeliveryArtifact, path: Path) -> dict[str, object]:
        return {
            "role": artifact.role.value,
            "filename": path.name,
            "path": str(path),
            "kind": artifact.kind,
            "timeline": artifact.timeline,
            "purpose": artifact.purpose,
            "source_role": artifact.source_role.value if artifact.source_role else None,
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() and path.is_file() else 0,
        }
