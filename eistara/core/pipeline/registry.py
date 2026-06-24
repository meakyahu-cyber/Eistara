from __future__ import annotations

from dataclasses import dataclass, field

from eistara.core.jobs.models import STAGE_ORDER, StageName

from .stage import StageContext, StageResult, StageRunner


@dataclass(frozen=True, slots=True)
class NoopStageRunner:
    stage: StageName

    def run(self, context: StageContext) -> StageResult:
        return StageResult(status="skipped", skipped=True, warnings=[f"Noop runner for {self.stage.value}"])


def noop_runners(stages: tuple[StageName, ...] | list[StageName] = STAGE_ORDER) -> list[NoopStageRunner]:
    return [NoopStageRunner(stage) for stage in stages]


@dataclass(slots=True)
class StageRegistry:
    runners: dict[StageName, StageRunner] = field(default_factory=dict)

    def register(self, runner: StageRunner) -> None:
        self.runners[runner.stage] = runner

    def get(self, stage: StageName) -> StageRunner | None:
        return self.runners.get(stage)

    def require(self, stage: StageName) -> StageRunner:
        runner = self.get(stage)
        if runner is None:
            raise KeyError(f"No runner registered for stage: {stage.value}")
        return runner

    def registered_stages(self) -> list[StageName]:
        stages = [stage for stage in STAGE_ORDER if stage in self.runners]
        stages.extend(stage for stage in self.runners if stage not in STAGE_ORDER)
        return stages

    def missing_stages(self) -> list[StageName]:
        return [stage for stage in STAGE_ORDER if stage not in self.runners]
