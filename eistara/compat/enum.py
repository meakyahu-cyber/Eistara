from __future__ import annotations

try:
    from enum import StrEnum as StrEnum
except ImportError:
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return str(self.value)
