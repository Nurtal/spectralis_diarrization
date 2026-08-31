import numpy as np
import pytest

from benchmark.separation_metrics import (
    best_pairing_si_sdr,
    bss_metrics,
    pesq_score,
    si_sdr,
    speech_quality_metrics,
    stoi_score,
)


def tone(freq, dur_s, sr=8000, amp=0.3):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestSiSdr:
    def test_perfect_reconstruction_very_high(self):
        ref = tone(440.0, 0.5)
        assert si_sdr(ref, ref.copy()) > 100.0

    def test_known_noise_level_matches_expected_si_sdr(self):
        rng = np.random.default_rng(0)
        ref = tone(440.0, 2.0)
        noise_power_ratio_db = 10.0
        noise = rng.normal(
            0.0,
            np.sqrt(np.mean(ref**2) / 10 ** (noise_power_ratio_db / 10)),
            size=len(ref),
        ).astype(np.float32)
        est = ref + noise
        measured = si_sdr(ref, est)
        assert measured == pytest.approx(noise_power_ratio_db, abs=0.3)

    def test_zero_mean_invariance(self):
        ref = tone(440.0, 0.5)
        est = ref + 5.0  # DC offset must not change SI-SDR
        # both cases hit the eps ceiling (~100 dB); allow 1 dB slack
        assert si_sdr(ref, est) == pytest.approx(si_sdr(ref, ref.copy()), abs=1.0)
        assert si_sdr(ref, est) > 90.0

    def test_scale_invariance(self):
        rng = np.random.default_rng(3)
        ref = tone(440.0, 0.5)
        noise = rng.normal(0, 0.01, len(ref)).astype(np.float32)
        assert si_sdr(ref, 3.7 * (ref + noise)) == pytest.approx(si_sdr(ref, ref + noise), abs=1e-3)


class TestBestPairing:
    def test_permutation_invariant(self):
        a, b = tone(300.0, 0.5), tone(1500.0, 0.5)
        refs, order = [a, b], [b, a]
        direct = best_pairing_si_sdr(refs, [a.copy(), b.copy()])
        swapped_order = best_pairing_si_sdr(refs, order)
        assert direct == pytest.approx(swapped_order)

    def test_good_separation_scores_high(self):
        a, b = tone(300.0, 0.5), tone(1500.0, 0.5)
        score = best_pairing_si_sdr([a, b], [a.copy(), b.copy()])
        assert score > 50.0

    def test_blended_outputs_score_much_lower(self):
        a, b = tone(300.0, 0.5), tone(1500.0, 0.5)
        clean = best_pairing_si_sdr([a, b], [a.copy(), b.copy()])
        blended = best_pairing_si_sdr([a, b], [(a + b).copy(), (a + b).copy()])
        assert blended < clean - 40.0


class TestBssMetrics:
    def test_perfect_estimates_give_high_sdr(self):
        a, b = tone(300.0, 0.5), tone(1500.0, 0.5)
        m = bss_metrics([a, b], [a.copy(), b.copy()])
        assert m["sdr"] > 60.0
        assert m["sir"] > 60.0
        assert m["sar"] > 60.0

    def test_interference_lowers_sir(self):
        # estimate A contaminated by B: SIR must drop hard, SAR stays high
        a, b = tone(300.0, 0.5), tone(1500.0, 0.5)
        clean = bss_metrics([a, b], [a.copy(), b.copy()])
        leaked = bss_metrics([a, b], [a + 0.3 * b, b.copy()])
        assert leaked["sir"] < clean["sir"] - 20.0
        assert leaked["sar"] > 40.0

    def test_artifacts_lower_sar(self):
        rng = np.random.default_rng(1)
        a = tone(300.0, 0.5)
        b = tone(1500.0, 0.5)
        clean = bss_metrics([a, b], [a.copy(), b.copy()])
        noisy = bss_metrics([a, b], [a + rng.normal(0, 0.05, len(a)).astype(np.float32), b.copy()])
        assert noisy["sar"] < clean["sar"] - 10.0

    def test_returns_all_keys(self):
        a = tone(300.0, 0.2)
        b = tone(1500.0, 0.2)
        m = bss_metrics([a, b], [a.copy(), b.copy()])
        assert set(m) == {"sdr", "sir", "sar"}


class TestSpeechQuality:
    def test_pesq_perfect_is_high(self):
        pytest.importorskip("pesq")
        ref = tone(440.0, 1.0, sr=16000)
        # PESQ WB 1.0–4.5, perfect copy ~4.5
        assert pesq_score(ref, ref.copy(), sample_rate=16000) > 4.0

    def test_stoi_perfect_is_one(self):
        pytest.importorskip("pystoi")
        ref = tone(440.0, 1.0, sr=16000)
        assert stoi_score(ref, ref.copy(), sample_rate=16000) == pytest.approx(1.0, abs=0.02)

    def test_quality_degrades_with_noise(self):
        pesq_available = pytest.importorskip("pesq", reason="pesq not installed")  # noqa: F841
        pytest.importorskip("pystoi")
        rng = np.random.default_rng(0)
        ref = tone(440.0, 1.0, sr=16000)
        noise = rng.normal(0, 0.05, len(ref)).astype(np.float32)
        noisy = (ref + noise).astype(np.float32)
        clean_pesq = pesq_score(ref, ref.copy(), sample_rate=16000)
        noisy_pesq = pesq_score(ref, noisy, sample_rate=16000)
        assert noisy_pesq < clean_pesq - 0.3
        clean_stoi = stoi_score(ref, ref.copy(), sample_rate=16000)
        noisy_stoi = stoi_score(ref, noisy, sample_rate=16000)
        assert noisy_stoi < clean_stoi - 0.05

    def test_speech_quality_metrics_averaged(self):
        pytest.importorskip("pesq")
        pytest.importorskip("pystoi")
        a = tone(300.0, 1.0, sr=16000)
        b = tone(1500.0, 1.0, sr=16000)
        m = speech_quality_metrics([a, b], [a.copy(), b.copy()], sample_rate=16000)
        assert "pesq" in m and "stoi" in m
        assert m["pesq"] > 4.0
        assert m["stoi"] == pytest.approx(1.0, abs=0.02)

    def test_speech_quality_fallback_without_libs(self, monkeypatch):
        # simulate missing libs via monkeypatching import to raise
        import benchmark.separation_metrics as mod

        monkeypatch.setattr(mod, "_pesq_available", lambda: False)
        monkeypatch.setattr(mod, "_stoi_available", lambda: False)
        a = tone(300.0, 0.5, sr=16000)
        m = speech_quality_metrics([a], [a.copy()], sample_rate=16000)
        assert m == {}

    def test_pesq_requires_16000_or_8000(self):
        ref = tone(440.0, 0.5, sr=16000)
        # wrapper documents wideband 16k only — ensure we either
        # resample or raise clearly; T2 enforces 16k so 8k must raise
        with pytest.raises((ValueError, RuntimeError)):
            pesq_score(ref, ref.copy(), sample_rate=8000)
