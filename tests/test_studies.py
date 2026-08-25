import numpy as np
import pytest

from benchmark.enrollment import degraded_enrollment, select_enrollment_segments
from benchmark.interfaces import Segment
from benchmark.studies import embedding_robustness_curve

SR = 8000


def tone(freq, dur_s, amp=0.3):
    t = np.arange(int(dur_s * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class BandEncoder:
    def encode(self, audio, sample_rate):
        from benchmark.stft import stft

        spec, params = stft(np.asarray(audio, dtype=np.float32), sample_rate)
        freqs = np.fft.rfftfreq(params.n_fft, 1 / sample_rate)
        low = float((np.abs(spec)[freqs < 1000] ** 2).sum())
        high = float((np.abs(spec)[freqs >= 1000] ** 2).sum())
        v = np.array([low, high], dtype=np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v


class TestEmbeddingRobustnessCurve:
    def test_similarity_degrades_with_noise(self):
        audio = np.concatenate([tone(200.0, 2.0), tone(200.0, 0.5)])
        spans = select_enrollment_segments([Segment(0.0, 2.5, "A")], speaker="A", max_duration=2.0)
        records = embedding_robustness_curve(
            BandEncoder(), audio, SR, spans, snrs=[None, 20, 10, 0], seed=0
        )
        assert len(records) == 4
        clean = next(r["cosine"] for r in records if r["snr"] is None)
        at_0 = next(r["cosine"] for r in records if r["snr"] == 0)
        assert clean == pytest.approx(1.0, abs=1e-5)
        assert at_0 < clean - 0.01

    def test_records_include_speaker_and_snr(self):
        audio = tone(300.0, 2.0)
        spans = [(0.0, 2.0)]
        records = embedding_robustness_curve(
            BandEncoder(), audio, SR, spans, snrs=[10], speaker="B"
        )
        assert records[0]["speaker"] == "B"
        assert records[0]["snr"] == 10


class TestDegradedEnrollment:
    def test_clean_passthrough(self):
        clip = tone(300.0, 1.0)
        out = degraded_enrollment(clip, SR, [(0.0, 1.0)], snr_db=None)
        np.testing.assert_array_equal(out, clip)

    def test_noisy_version_differs_but_same_length(self):
        clip = tone(300.0, 1.0)
        out = degraded_enrollment(clip, SR, [(0.0, 1.0)], snr_db=5, seed=1)
        assert len(out) == len(clip)
        assert not np.array_equal(out, clip)
