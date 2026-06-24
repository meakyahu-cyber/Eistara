from __future__ import annotations

from eistara.core.delivery import SubtitleRow
from eistara.core.dubbing import AudioMixPlan
from eistara.core.subtitle import subtitle_visible_len
from eistara.core.timeline import DubTimeline
from eistara.core.translation.validator import has_excess_latin_text

from .models import QualityIssue, QualitySeverity


def check_translations(
    translations: dict[int, str],
    *,
    target_language: str = "Simplified Chinese",
    allowed_latin_by_id: dict[int, set[str]] | None = None,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for item_id, text in sorted(translations.items()):
        if not str(text).strip():
            issues.append(
                QualityIssue(
                    code="translation.empty",
                    message="Translation is empty",
                    severity=QualitySeverity.ERROR,
                    location=f"translation:{item_id}",
                )
            )
            continue
        allowed = (allowed_latin_by_id or {}).get(item_id, set())
        if has_excess_latin_text(text, allowed, target_language=target_language):
            issues.append(
                QualityIssue(
                    code="translation.latin_residue",
                    message="Translation appears to contain untranslated Latin text",
                    severity=QualitySeverity.ERROR,
                    location=f"translation:{item_id}",
                    details={"text": str(text)[:160]},
                )
            )
    return issues


def check_subtitle_rows(
    rows: list[SubtitleRow],
    *,
    max_source_chars: int = 42,
    max_target_chars: int = 24,
    min_duration_sec: float = 0.15,
) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    previous_end = 0.0
    for index, row in enumerate(rows, 1):
        location = f"subtitle:{index}"
        if row.end_sec <= row.start_sec:
            issues.append(QualityIssue("subtitle.invalid_time", "Subtitle end must be after start", QualitySeverity.ERROR, location))
            continue
        if row.end_sec - row.start_sec < min_duration_sec:
            issues.append(QualityIssue("subtitle.too_short", "Subtitle duration is very short", QualitySeverity.WARNING, location))
        if row.start_sec < previous_end:
            issues.append(QualityIssue("subtitle.overlap", "Subtitle overlaps previous row", QualitySeverity.WARNING, location))
        if subtitle_visible_len(row.source) > max_source_chars:
            issues.append(QualityIssue("subtitle.source_too_long", "Source subtitle text is long", QualitySeverity.WARNING, location))
        if subtitle_visible_len(row.target) > max_target_chars:
            issues.append(QualityIssue("subtitle.target_too_long", "Target subtitle text is long", QualitySeverity.WARNING, location))
        previous_end = max(previous_end, row.end_sec)
    return issues


def check_timeline(timeline: DubTimeline) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for warning in timeline.warnings:
        issues.append(QualityIssue("timeline.warning", warning, QualitySeverity.WARNING, "timeline"))
    for segment in timeline.segments:
        location = f"timeline:{segment.segment_id}"
        if segment.audio_path is None:
            issues.append(QualityIssue("timeline.missing_audio_path", "Timeline segment has no audio path", QualitySeverity.ERROR, location))
        if segment.audio_duration_sec <= 0:
            issues.append(QualityIssue("timeline.empty_audio", "Timeline segment has empty audio duration", QualitySeverity.ERROR, location))
        if segment.dub_end_sec <= segment.dub_start_sec:
            issues.append(QualityIssue("timeline.invalid_time", "Timeline segment end must be after start", QualitySeverity.ERROR, location))
    return issues


def check_audio_mix_plan(plan: AudioMixPlan) -> list[QualityIssue]:
    issues: list[QualityIssue] = []
    for warning in plan.warnings:
        issues.append(QualityIssue("audio_mix.warning", warning, QualitySeverity.WARNING, "audio_mix"))
    if not plan.clips:
        issues.append(QualityIssue("audio_mix.no_clips", "Audio mix plan has no clips", QualitySeverity.ERROR, "audio_mix"))
    for clip in plan.clips:
        location = f"audio_mix:{clip.segment_id}"
        if clip.duration_sec <= 0:
            issues.append(QualityIssue("audio_mix.invalid_clip_time", "Audio clip end must be after start", QualitySeverity.ERROR, location))
    return issues
