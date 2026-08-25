import importlib.util

import numpy as np
import pytest

from benchmark.registry import SEPARATORS


def speechbrain_available():
    try:
        return importlib.util.find_spec("speechbrain") is not None
    except (ImportError, ValueError):
        return False


class FakeModel:
    """Duck-typed SpeechBrain model returning two fixed sources at 8 kHz.

    Output length mirrors the input waveform length (as real models do).
    """

    sample_rate = 8000

    def separate_batch(self, waveform):
        wave = np.asarray(waveform)
        n_out = wave.shape[-1]
        t = np.arange(n_out) / self.sample_rate
        low = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
        high = (0.3 * np.sin(2 * np.pi * 2400 * t)).astype(np.float32)
        return np.stack([low, high], axis=-1)[None]


class TestSpeechBrainSeparator:
    def test_registered_under_model_names(self):
        for name in ("sepformer", "conv_tasnet", "dprnn"):
            assert name in SEPARATORS

    @pytest.mark.skipif(speechbrain_available(), reason="speechbrain installed")
    def test_helpful_error_when_missing(self):
        with pytest.raises(RuntimeError, match="pip install"):
            SEPARATORS["sepformer"]()

    def test_injected_model_produces_sources_at_input_rate(self):
        sep = SEPARATORS["sepformer"](model=FakeModel())
        sr = 16000
        audio = np.zeros(sr // 2, dtype=np.float32)  # 0.5 s at 16 kHz
        sources = sep.separate(audio, sr)

        assert len(sources) == 2
        # model outputs at 8 kHz; adapter must resample back to input rate/length
        assert all(abs(len(s) - len(audio)) <= 2 for s in sources)

    def test_output_is_float_and_finite(self):
        sep = SEPARATORS["sepformer"](model=FakeModel())
        sources = sep.separate(np.zeros(4000, dtype=np.float32), 16000)
        for s in sources:
            assert s.dtype == np.float32
            assert np.isfinite(s).all()

    @pytest.mark.skipif(not speechbrain_available(), reason="speechbrain not installed")
    def test_real_model_smoke(self):
        pytest.importorskip("torch")
        sep = SEPARATORS["sepformer"]()
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.1, 0.1, size=32000).astype(np.float32)  # 2 s
        sources = sep.separate(audio, 16000, num_speakers=2)
        assert len(sources) == 2
