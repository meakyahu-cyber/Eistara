from __future__ import annotations

from eistara.core.pipeline import StageResult

from .protocol import StageDiagnosticContext


class NoopDiagnosticsHook:
    def on_stage_finished(self, context: StageDiagnosticContext, result: StageResult) -> None:
        return None

    def on_stage_failed(self, context: StageDiagnosticContext, error: str, result: StageResult | None = None) -> None:
        return None
