"""Model registries. Ships VAD baseline; pyannote registered when installed."""

from benchmark.interfaces import Diarizer, Separator
from benchmark.neural_separator import SpeechBrainSeparator
from benchmark.nmf_separator import NMFSeparator
from benchmark.pyannote_diarizer import PyannoteDiarizer
from benchmark.speaker_encoder import SpeakerEncoder
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
    "nmf": NMFSeparator,
    # Neural separators (ADR-004: pretrained checkpoints). Force CPU due to old CUDA driver.
    "sepformer": lambda **kw: SpeechBrainSeparator(
        model_id=kw.pop("model_id", "speechbrain/sepformer-wsj02mix"),
        model_class=kw.pop("model_class", "sepformer"),
        device=kw.pop("device", "cpu"),
        **kw,
    ),
    "sepformer3": lambda **kw: SpeechBrainSeparator(
        model_id=kw.pop("model_id", "speechbrain/sepformer-wsj03mix"),
        model_class=kw.pop("model_class", "sepformer"),
        device=kw.pop("device", "cpu"),
        **kw,
    ),
    "conv_tasnet": lambda **kw: SpeechBrainSeparator(
        model_id=kw.pop("model_id", "speechbrain/ConvTasNet-Wham"),
        model_class=kw.pop("model_class", "sepformer"),
        device=kw.pop("device", "cpu"),
        **kw,
    ),
    "dprnn": lambda **kw: SpeechBrainSeparator(
        model_id=kw.pop("model_id", "speechbrain/dprnn-wsj02mix"),
        model_class=kw.pop("model_class", "sepformer"),
        device=kw.pop("device", "cpu"),
        **kw,
    ),
    "tf_gridnet": lambda **kw: SpeechBrainSeparator(
        model_id=kw.pop("model_id", "espnet/tf_gridnet_wsj0_2mix"),
        model_class=kw.pop("model_class", "tfgridnet"),
        device=kw.pop("device", "cpu"),
        **kw,
    ),
    # alias without underscore
    "tfgridnet": lambda **kw: SpeechBrainSeparator(
        model_id=kw.pop("model_id", "espnet/tf_gridnet_wsj0_2mix"),
        model_class=kw.pop("model_class", "tfgridnet"),
        device=kw.pop("device", "cpu"),
        **kw,
    ),
}

DIARIZERS = {
    "noop": NoopDiarizer,
    "vad": EnergyVadDiarizer,
    # Constructor raises an informative RuntimeError when pyannote.audio or
    # the HuggingFace token are unavailable.
    "pyannote": PyannoteDiarizer,
}

ENCODERS = {
    "ecapa": SpeakerEncoder,
}
