import importlib.util

import numpy as np
import pytest

from benchmark.registry import ENCODERS


def speechbrain_available():
    try:
        return importlib.util.find_spec("speechbrain") is not None
    except (ImportError, ValueError):
        return False


class FakeEncoder:
    """Returns a fixed embedding regardless of input."""

    def __init__(self, dim=8):
        self.dim = dim

    def encode(self, audio, sample_rate):
        v = np.ones(self.dim, dtype=np.float32)
        return v / np.linalg.norm(v)


class TestSpeakerEncoder:
    def test_ecapa_registered(self):
        assert "ecapa" in ENCODERS

    @pytest.mark.skipif(speechbrain_available(), reason="speechbrain installed")
    def test_helpful_error_when_missing(self):
        with pytest.raises(RuntimeError, match="pip install"):
            ENCODERS["ecapa"]()

    def test_injected_encoder_returns_unit_vector(self):
        enc = ENCODERS["ecapa"](model=FakeEncoder())
        emb = enc.encode(np.zeros(4000, dtype=np.float32), 16000)
        assert emb.shape == (8,)
        assert emb.dtype == np.float32
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_embeddings_of_same_audio_are_identical(self):
        enc = ENCODERS["ecapa"](model=FakeEncoder())
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.1, 0.1, 8000).astype(np.float32)
        np.testing.assert_array_equal(enc.encode(audio, 16000), enc.encode(audio, 16000))

    @pytest.mark.skipif(not speechbrain_available(), reason="speechbrain not installed")
    def test_real_ecapa_smoke(self):
        enc = ENCODERS["ecapa"]()
        rng = np.random.default_rng(0)
        audio = rng.uniform(-0.1, 0.1, 32000).astype(np.float32)
        emb = enc.encode(audio, 16000)
        assert emb.ndim == 1 and len(emb) >= 64


class TestEnrollmentPolicy:
    def test_picks_longest_clean_solo_spans(self):
        from benchmark.enrollment import select_enrollment_segments
        from benchmark.interfaces import Segment

        segments = [
            Segment(0.0, 3.0, "A"),
            Segment(2.0, 4.0, "B"),  # overlap with A on [2,3]
            Segment(4.5, 6.0, "A"),
        ]
        spans = select_enrollment_segments(segments, speaker="A", max_duration=60.0)
        # solo A spans: [0,2] and [3,4] merged? no: [0,2], [3,4], [4.5,6]
        covered = sum(e - s for s, e in spans)
        assert covered > 2.5
        for s, e in spans:
            assert e > s

    def test_respects_max_duration(self):
        from benchmark.enrollment import select_enrollment_segments
        from benchmark.interfaces import Segment

        segments = [Segment(0.0, 10.0, "A")]
        spans = select_enrollment_segments(segments, speaker="A", max_duration=2.0)
        assert sum(e - s for s, e in spans) <= 2.0 + 1e-6

    def test_no_speech_for_unknown_speaker(self):
        from benchmark.enrollment import select_enrollment_segments
        from benchmark.interfaces import Segment

        assert select_enrollment_segments([Segment(0, 1, "A")], speaker="Z") == []

    def test_loads_enrollment_audio_from_mixture(self):
        from pathlib import Path

        from benchmark.audio import load_audio, save_audio
        from benchmark.enrollment import enrollment_audio

        sr = 8000
        path = Path("/tmp/opencode/enroll_test.wav")
        path.parent.mkdir(parents=True, exist_ok=True)
        t = np.arange(int(2 * sr)) / sr
        save_audio(path, (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), sr)

        audio, sr_out = load_audio(path)
        clip = enrollment_audio(audio, sr_out, [(0.0, 1.0)])
        assert abs(len(clip) - sr_out) <= 2
