"""Model registries. Ships VAD baseline; pyannote registered when installed."""

from benchmark.interfaces import Diarizer, Separator
from benchmark.pyannote_diarizer import PyannoteDiarizer
from benchmark.vad import energy_vad


class IdentitySeparator(Separator):
    """Returns the input as a single source. Placeholder for real separators."""

    def separate(self, audio, sample_rate, num_speakers=None):
        audio = self._validate_audio(audio, sample_rate)
        return [audio.copy()]


class NoopDiarizer(Diarizer):
    """Detects no speech at all. Placeholder for real diarizers."""

    def diarize(self, audio, sample_rate):
        self._validate_audio(audio, sample_rate)
        return []


class EnergyVadDiarizer(Diarizer):
    """Energy-based single-label diarizer: marks speech vs silence (no speakers)."""

    def __init__(self, threshold_db=-40.0, min_speech_s=0.1):
        self.threshold_db = threshold_db
        self.min_speech_s = min_speech_s

    def diarize(self, audio, sample_rate):
        audio = self._validate_audio(audio, sample_rate)
        return energy_vad(
            audio,
            sample_rate,
            threshold_db=self.threshold_db,
            min_speech_s=self.min_speech_s,
        )


SEPARATORS = {
    "identity": IdentitySeparator,
}

DIARIZERS = {
    "noop": NoopDiarizer,
    "vad": EnergyVadDiarizer,
    # Constructor raises an informative RuntimeError when pyannote.audio or
    # the HuggingFace token are unavailable.
    "pyannote": PyannoteDiarizer,
}
