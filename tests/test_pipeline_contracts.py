from __future__ import annotations

from pathlib import Path

from eistara.core.jobs.models import STAGE_ORDER, StageName
from eistara.core.pipeline import DEFAULT_ARTIFACT_CONTRACTS, StageContext, StageRegistry, StageResult


class RecordingStageRunner:
    def __init__(self, stage: StageName):
        self.stage = stage

    def run(self, context: StageContext) -> StageResult:
        return StageResult(outputs={"stage": context.stage.value})


def test_artifact_contract_reports_missing_and_existing_outputs(tmp_path: Path) -> None:
    contract = DEFAULT_ARTIFACT_CONTRACTS[StageName.COMPOSE]

    assert contract.is_satisfied(tmp_path) is False
    assert [item.name for item in contract.missing_required(tmp_path)] == ["dub_video"]

    output = tmp_path / "output" / "output_dub.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"demo")

    assert contract.is_satisfied(tmp_path) is True
    assert contract.existing_outputs(tmp_path)["dub_video"] == str(output)


def test_stage_registry_tracks_registered_and_missing_stages() -> None:
    registry = StageRegistry()
    registry.register(RecordingStageRunner(StageName.DOWNLOAD))

    assert registry.registered_stages() == [StageName.DOWNLOAD]
    assert StageName.TRANSLATE in registry.missing_stages()


def test_stage_registry_can_cover_requested_stages() -> None:
    registry = StageRegistry()
    for stage in STAGE_ORDER:
        registry.register(RecordingStageRunner(stage))

    assert registry.registered_stages() == list(STAGE_ORDER)
