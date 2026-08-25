"""Model registries. Phase 0 ships only dummy implementations."""

from benchmark.interfaces import Diarizer, Separator


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


SEPARATORS = {
    "identity": IdentitySeparator,
}

DIARIZERS = {
    "noop": NoopDiarizer,
}
