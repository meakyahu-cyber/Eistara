"""ASR provider adapters."""

from .demucs import DemucsVocalSeparationProvider
from .v1_whisper import WhisperRuntimeAsrProvider

__all__ = [
    "DemucsVocalSeparationProvider",
    "WhisperRuntimeAsrProvider",
]
