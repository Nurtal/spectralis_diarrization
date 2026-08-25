import numpy as np
import pytest

from benchmark.interfaces import Segment  # noqa: F401 (interface sanity)
from benchmark.nmf_separator import NMFSeparator

SR = 8000


def tone(freq, dur_s, sr=SR, amp=0.3):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def disjoint_mixture(dur_s=2.0):
    """Two speakers with well-separated bands and turn-taking activity.

    Simultaneous stationary tones in *overlapping* bands are a pathological
    degenerate case for blind NMF; real speakers differ in temporal activity,
    which is what NMF exploits.
    """
    low = tone(200.0, dur_s / 2)
    high_start = int(0.75 * len(low))
    high = np.zeros(int(dur_s * SR), dtype=np.float32)
    high[high_start:] = tone(2400.0, dur_s - 0.75 * dur_s / 2)[: len(high) - high_start]
    padded_low = np.zeros(int(dur_s * SR), dtype=np.float32)
    padded_low[: len(low)] = low
    mixture = np.clip(padded_low + high, -1.0, 1.0)
    return mixture, [padded_low, high]


class TestNMFSeparator:
    def test_implements_separator_interface(self):
        sep = NMFSeparator(num_speakers=2)
        from benchmark.interfaces import Separator

        assert isinstance(sep, Separator)

    def test_returns_requested_number_of_sources(self):
        mixture, _ = disjoint_mixture()
        sources = NMFSeparator(num_speakers=2, seed=0).separate(mixture, SR)
        assert len(sources) == 2
        assert all(len(s) == len(mixture) for s in sources)

    def test_separates_disjoint_band_speakers(self):
        mixture, refs = disjoint_mixture()
        sources = NMFSeparator(num_speakers=2, seed=0).separate(mixture, SR)

        from benchmark.separation_metrics import best_pairing_si_sdr

        score = best_pairing_si_sdr(refs, sources)
        assert score > 5.0, f"SI-SDR too low for disjoint bands: {score:.2f} dB"

    def test_deterministic_given_seed(self):
        mixture, _ = disjoint_mixture()
        s1 = NMFSeparator(num_speakers=2, seed=7).separate(mixture, SR)
        s2 = NMFSeparator(num_speakers=2, seed=7).separate(mixture, SR)
        for a, b in zip(s1, s2):
            np.testing.assert_array_equal(a, b)

    def test_rejects_non_positive_num_speakers(self):
        mixture, _ = disjoint_mixture(dur_s=0.5)
        with pytest.raises(ValueError, match="num_speakers"):
            NMFSeparator(num_speakers=0).separate(mixture, SR)

    def test_registered_in_registry(self):
        from benchmark.registry import SEPARATORS

        assert "nmf" in SEPARATORS
