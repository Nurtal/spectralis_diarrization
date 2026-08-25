import numpy as np

from benchmark.stft import istft, stft

SR = 16000


def sine(freq, dur_s, sr=SR, amp=0.5):
    t = np.arange(int(dur_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestStftRoundtrip:
    def test_roundtrip_reconstructs_signal(self):
        x = sine(440.0, 1.0)
        spec, params = stft(x, sample_rate=SR)
        y, _ = istft(spec, params, length=len(x))
        assert len(y) == len(x)
        # interior samples (away from edges) reconstructed near-perfectly
        np.testing.assert_allclose(y[2048:-2048], x[2048:-2048], atol=1e-4)

    def test_frequency_peak_found(self):
        x = sine(1000.0, 0.5)
        spec, params = stft(x, sample_rate=SR)
        mean_mag = np.abs(spec).mean(axis=1)
        peak_bin = int(np.argmax(mean_mag))
        freqs = np.fft.rfftfreq(params.n_fft, 1 / SR)
        assert abs(freqs[peak_bin] - 1000.0) < 50.0
