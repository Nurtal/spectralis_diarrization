from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Segment:
    """A labeled time span: speaker `speaker` is active in [start, end) seconds."""

    start: float
    end: float
    speaker: str

    def __post_init__(self):
        if self.end < self.start:
            raise ValueError(f"Segment end ({self.end}) must not be before start ({self.start})")


class Separator(ABC):
    """Common interface for all speech separation approaches (ADR-003)."""

    @abstractmethod
    def separate(self, audio, sample_rate, num_speakers=None):
        """Separate `audio` into per-source signals.

        Returns a list of np.ndarray waveforms, one per estimated source.
        `num_speakers` may be None when the approach estimates it.
        """
        ...

    @staticmethod
    def _validate_audio(audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be a 1D mono waveform")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return audio


class Diarizer(ABC):
    """Common interface for all diarization approaches (ADR-003)."""

    @abstractmethod
    def diarize(self, audio, sample_rate):
        """Return the list of Segments describing who speaks when."""
        ...

    @staticmethod
    def _validate_audio(audio, sample_rate):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError("audio must be a 1D mono waveform")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return audio
