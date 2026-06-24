from __future__ import annotations

from dataclasses import dataclass

from eistara.core.delivery import SubtitleRow

from .models import AsrRequest, AsrResult, AsrSegment, AsrSettings
from .providers import AsrProvider


@dataclass(slots=True)
class AsrService:
    provider: AsrProvider
    settings: AsrSettings = AsrSettings()

    def transcribe(self, request: AsrRequest) -> AsrResult:
        raw = self.provider.transcribe(request, self.settings)
        segments, warnings = normalize_asr_segments(raw.segments)
        return AsrResult(
            segments=tuple(segments),
            language=raw.language or request.language or self.settings.language,
            warnings=tuple([*raw.warnings, *warnings]),
        )


def normalize_asr_segments(segments: tuple[AsrSegment, ...] | list[AsrSegment]) -> tuple[list[AsrSegment], list[str]]:
    normalized: list[AsrSegment] = []
    warnings: list[str] = []
    previous_end = 0.0
    next_id = 1

    for segment in sorted(segments, key=lambda item: (item.start_sec, item.end_sec, item.id)):
        text = " ".join(segment.text.split())
        if not text:
            warnings.append(f"{segment.id}: skipped empty text")
            continue
        start = max(0.0, float(segment.start_sec))
        end = max(start, float(segment.end_sec))
        if end == start:
            warnings.append(f"{segment.id}: skipped zero-length segment")
            continue
        if start < previous_end:
            warnings.append(f"{segment.id}: adjusted overlap from {start:.3f} to {previous_end:.3f}")
            start = previous_end
            end = max(end, start)
        if end == start:
            warnings.append(f"{segment.id}: skipped after overlap adjustment")
            continue
        normalized.append(
            AsrSegment(
                id=next_id,
                start_sec=start,
                end_sec=end,
                text=text,
                speaker=_speaker_id(segment.speaker),
                words=segment.words,
            )
        )
        previous_end = end
        next_id += 1

    return normalized, warnings


def asr_segments_to_subtitle_rows(segments: tuple[AsrSegment, ...] | list[AsrSegment]) -> list[SubtitleRow]:
    return [
        SubtitleRow(
            start_sec=segment.start_sec,
            end_sec=segment.end_sec,
            source=segment.text,
            target="",
            speaker=_speaker_id(segment.speaker),
        )
        for segment in segments
    ]


def _speaker_id(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "SPEAKER_00"
