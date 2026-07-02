from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess
from typing import Protocol

from eistara.core.media import (
    AudioExtractPlan,
    ComposeVideoPlan,
    MediaCommandResult,
    MediaInfo,
    MediaProbeError,
)


class ProcessRunner(Protocol):
    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        """Run a process and return a completed process."""


class FfmpegProcessRunner:
    def run(self, args: tuple[str, ...]) -> CompletedProcess[str]:
        import subprocess

        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


@dataclass(slots=True)
class FfmpegMediaProvider:
    runner: ProcessRunner | None = None
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    name: str = "ffmpeg"

    def __post_init__(self) -> None:
        if self.runner is None:
            self.runner = FfmpegProcessRunner()

    def probe(self, path: str) -> MediaInfo:
        args = (
            self.ffprobe_path,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(Path(path)),
        )
        result = self.runner.run(args)
        if result.returncode != 0:
            raise MediaProbeError((result.stderr or result.stdout or "ffprobe failed").strip())
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise MediaProbeError(f"ffprobe returned invalid JSON: {exc}") from exc
        return MediaInfo.from_ffprobe(path, data)

    def extract_audio(self, plan: AudioExtractPlan) -> MediaCommandResult:
        plan.output_audio.parent.mkdir(parents=True, exist_ok=True)
        return self._run_command(plan.ffmpeg_args(self.ffmpeg_path))

    def compose_video(self, plan: ComposeVideoPlan) -> MediaCommandResult:
        plan.output_video.parent.mkdir(parents=True, exist_ok=True)
        return self._run_command(plan.ffmpeg_args(self.ffmpeg_path))

    def _run_command(self, args: tuple[str, ...]) -> MediaCommandResult:
        result = self.runner.run(args)
        return MediaCommandResult(
            command=args,
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )
