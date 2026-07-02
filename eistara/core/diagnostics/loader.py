from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

from eistara.core.pipeline import StageResult

from .noop import NoopDiagnosticsHook
from .protocol import DiagnosticsHook, StageDiagnosticContext


MODULE_ENV = "EISTARA_DIAGNOSTICS_MODULE"
PATH_ENV = "EISTARA_DIAGNOSTICS_PATH"

_HOOK: DiagnosticsHook | None = None


def get_diagnostics_hook() -> DiagnosticsHook:
    global _HOOK
    if _HOOK is None:
        _HOOK = _load_hook()
    return _HOOK


def notify_stage_finished(context: StageDiagnosticContext, result: StageResult) -> None:
    try:
        get_diagnostics_hook().on_stage_finished(context, result)
    except Exception:
        return None


def notify_stage_failed(context: StageDiagnosticContext, error: str, result: StageResult | None = None) -> None:
    try:
        get_diagnostics_hook().on_stage_failed(context, error, result)
    except Exception:
        return None


def _load_hook() -> DiagnosticsHook:
    module_name = os.environ.get(MODULE_ENV, "").strip()
    if not module_name:
        return NoopDiagnosticsHook()
    _add_local_diagnostics_paths()
    try:
        module = importlib.import_module(module_name)
        hook = _hook_from_module(module)
        return hook if hook is not None else NoopDiagnosticsHook()
    except Exception:
        return NoopDiagnosticsHook()


def _add_local_diagnostics_paths() -> None:
    candidates = [os.environ.get(PATH_ENV, "").strip(), str(Path.cwd() / ".local_diagnostics")]
    for item in candidates:
        if not item:
            continue
        path = str(Path(item).expanduser().resolve())
        if Path(path).exists() and path not in sys.path:
            sys.path.insert(0, path)


def _hook_from_module(module: Any) -> DiagnosticsHook | None:
    create_hook = getattr(module, "create_hook", None)
    if callable(create_hook):
        return create_hook()
    hook = getattr(module, "hook", None)
    if hook is not None:
        return hook
    if hasattr(module, "on_stage_finished") or hasattr(module, "on_stage_failed"):
        return _ModuleFunctionHook(module)
    return None


class _ModuleFunctionHook:
    def __init__(self, module: Any):
        self.module = module

    def on_stage_finished(self, context: StageDiagnosticContext, result: StageResult) -> None:
        callback = getattr(self.module, "on_stage_finished", None)
        if callable(callback):
            callback(context, result)

    def on_stage_failed(self, context: StageDiagnosticContext, error: str, result: StageResult | None = None) -> None:
        callback = getattr(self.module, "on_stage_failed", None)
        if callable(callback):
            callback(context, error, result)
