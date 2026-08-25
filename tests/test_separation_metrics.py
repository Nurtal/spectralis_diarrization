import numpy as np
import pytest

from benchmark.separation_metrics import best_pairing_si_sdr, bss_metrics, si_sdr


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
