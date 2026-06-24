from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from eistara.compat.enum import StrEnum


class QualitySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    code: str
    message: str
    severity: QualitySeverity = QualitySeverity.WARNING
    location: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "location": self.location,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class QualityReport:
    issues: tuple[QualityIssue, ...] = ()

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == QualitySeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == QualitySeverity.WARNING)

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def extend(self, issues: list[QualityIssue] | tuple[QualityIssue, ...]) -> "QualityReport":
        return QualityReport((*self.issues, *issues))

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }
