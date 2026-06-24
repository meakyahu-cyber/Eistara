from __future__ import annotations

from dataclasses import dataclass

from eistara.core.delivery import SubtitleRow
from eistara.core.dubbing import AudioMixPlan
from eistara.core.timeline import DubTimeline

from .checks import check_audio_mix_plan, check_subtitle_rows, check_timeline, check_translations
from .models import QualityReport


@dataclass(frozen=True, slots=True)
class QualityGateService:
    target_language: str = "Simplified Chinese"
    max_source_chars: int = 42
    max_target_chars: int = 24

    def check(
        self,
        *,
        translations: dict[int, str] | None = None,
        subtitle_rows: list[SubtitleRow] | None = None,
        timeline: DubTimeline | None = None,
        audio_mix_plan: AudioMixPlan | None = None,
    ) -> QualityReport:
        report = QualityReport()
        if translations is not None:
            report = report.extend(check_translations(translations, target_language=self.target_language))
        if subtitle_rows is not None:
            report = report.extend(
                check_subtitle_rows(
                    subtitle_rows,
                    max_source_chars=self.max_source_chars,
                    max_target_chars=self.max_target_chars,
                )
            )
        if timeline is not None:
            report = report.extend(check_timeline(timeline))
        if audio_mix_plan is not None:
            report = report.extend(check_audio_mix_plan(audio_mix_plan))
        return report
