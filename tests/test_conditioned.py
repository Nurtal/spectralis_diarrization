import numpy as np
import pytest

from benchmark.conditioned_separator import ConditionedSeparator
from benchmark.registry import CONDITIONED_SEPARATORS


def tone(freq, dur_s=0.5, sr=8000, amp=0.3):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeEncoder:
    """Returns 2-D embedding distinguishing low vs high frequency."""

    def encode(self, audio, sr):
        # simple spectral centroid proxy: low freq -> [1,0], high -> [0,1]
        audio = np.asarray(audio, dtype=np.float32)
        # estimate dominant freq via correlation with tones
        low = tone(200, len(audio) / 8000, sr=8000)
        # trim to same length
        n = min(len(audio), len(low))
        corr_low = float(np.dot(audio[:n], low[:n]))
        # high tone
        high = tone(2000, len(audio) / 8000, sr=8000)
        corr_high = float(np.dot(audio[:n], high[:n]))
        vec = np.array([corr_low, corr_high], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec


class FakeSeparator:
    """Blind separator returning two fixed tones."""

    def separate(self, audio, sr, num_speakers=None):
        # ignore input, return low and high tones at input sr/len
        sr_out = sr
        dur = len(audio) / sr_out
        return [tone(200, dur, sr=sr_out), tone(2000, dur, sr=sr_out)]


class FakeTargetModel:
    """Injected conditioned model with separate_target."""

    def separate_target(self, mixture, enrollment):
        # return enrollment as target for testing
        return enrollment.copy()


class TestConditionedRegistry:
    def test_registered(self):
        for name in ("spexplus", "conditioned_nmf", "conditioned_sepformer"):
            assert name in CONDITIONED_SEPARATORS


class TestConditionedFallback:
    def test_separate_then_select_picks_correct_speaker(self):
        sep = FakeSeparator()
        enc = FakeEncoder()
        cond = ConditionedSeparator(separator=sep, encoder=enc, model_class="fallback")
        rng = np.random.default_rng(0)
        mixture = rng.uniform(-0.1, 0.1, 8000).astype(np.float32)
        # enrollment is low tone -> should pick low source
        enroll_low = tone(200, 0.5, sr=8000)
        target = cond.extract_target(mixture, 8000, enroll_low, 8000)
        # target should be low tone (dominant low freq)
        # check via encoder cosine
        emb_target = enc.encode(target, 8000)
        emb_low = enc.encode(enroll_low, 8000)
        assert float(np.dot(emb_target, emb_low)) > 0.9

        # enrollment high -> pick high
        enroll_high = tone(2000, 0.5, sr=8000)
        target2 = cond.extract_target(mixture, 8000, enroll_high, 8000)
        emb_target2 = enc.encode(target2, 8000)
        emb_high = enc.encode(enroll_high, 8000)
        assert float(np.dot(emb_target2, emb_high)) > 0.9

    def test_injected_model(self):
        cond = ConditionedSeparator(model=FakeTargetModel(), model_class="spexplus")
        mix = np.zeros(8000, dtype=np.float32)
        enrol = tone(200, 0.5, sr=8000)
        out = cond.extract_target(mix, 8000, enrol, 8000)
        np.testing.assert_allclose(out, enrol)

    def test_short_enrollment_still_works(self):
        sep = FakeSeparator()
        enc = FakeEncoder()
        cond = ConditionedSeparator(separator=sep, encoder=enc, model_class="fallback")
        mixture = np.zeros(8000, dtype=np.float32)
        short = tone(200, 0.1, sr=8000)  # 100ms
        target = cond.extract_target(mixture, 8000, short, 8000)
        assert target.dtype == np.float32
        assert len(target) == len(mixture)

    def test_empty_enrollment_raises(self):
        cond = ConditionedSeparator(
            separator=FakeSeparator(), encoder=FakeEncoder(), model_class="fallback"
        )
        with pytest.raises(ValueError, match="empty"):
            cond.extract_target(
                np.zeros(100, dtype=np.float32), 8000, np.zeros(0, dtype=np.float32), 8000
            )

    def test_separate_alias_returns_list(self):
        cond = ConditionedSeparator(
            separator=FakeSeparator(), encoder=FakeEncoder(), model_class="fallback"
        )
        enrol = tone(200, 0.5, sr=8000)
        out = cond.separate(np.zeros(8000, dtype=np.float32), 8000, enrol, 8000)
        assert isinstance(out, list) and len(out) == 1

    def test_missing_enrollment_raises(self):
        cond = ConditionedSeparator(
            separator=FakeSeparator(), encoder=FakeEncoder(), model_class="fallback"
        )
        with pytest.raises(ValueError, match="enrollment"):
            cond.separate(np.zeros(100, dtype=np.float32), 8000, None)

    def test_fallback_without_asteroid(self, monkeypatch):
        # simulate asteroid not installed but fallback still works
        import benchmark.conditioned_separator as mod

        monkeypatch.setattr(mod, "_asteroid_available", lambda: False)
        cond = ConditionedSeparator(
            separator=FakeSeparator(), encoder=FakeEncoder(), model_class="spexplus"
        )
        # should have fallen through to fallback
        assert cond._mode == "fallback"
        enrol = tone(200, 0.5, sr=8000)
        target = cond.extract_target(np.zeros(8000, dtype=np.float32), 8000, enrol, 8000)
        assert len(target) > 0
