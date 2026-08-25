import numpy as np
import pytest
import soundfile as sf

from benchmark.audio import load_audio, save_audio


def write_wav(tmp_path, data, sr):
    path = tmp_path / "clip.wav"
    sf.write(path, data, sr, subtype="FLOAT")
    return path


class TestLoadAudio:
    def test_roundtrip_preserves_mono_samples(self, tmp_path):
        sr = 16000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = write_wav(tmp_path, data, sr)

        audio, out_sr = load_audio(path)

        assert out_sr == sr
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert len(audio) == sr
        np.testing.assert_allclose(audio, data.astype(np.float32), atol=1e-6)

    def test_resamples_when_target_rate_given(self, tmp_path):
        sr_in = 44100
        data = np.zeros(sr_in, dtype=np.float32)
        path = write_wav(tmp_path, data, sr_in)

        audio, out_sr = load_audio(path, sample_rate=16000)

        assert out_sr == 16000
        expected_len = int(round(len(data) * 16000 / sr_in))
        assert abs(len(audio) - expected_len) <= 2

    def test_stereo_returns_two_columns(self, tmp_path):
        sr = 8000
        data = np.random.default_rng(0).uniform(-0.1, 0.1, size=(sr, 2)).astype(np.float32)
        path = write_wav(tmp_path, data, sr)

        audio, out_sr = load_audio(path)

        assert out_sr == sr
        assert audio.shape == (sr, 2)


class TestSaveAudio:
    def test_save_then_load_roundtrip(self, tmp_path):
        sr = 16000
        audio = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
        out = tmp_path / "out.wav"

        save_audio(out, audio, sr)

        loaded, loaded_sr = load_audio(out)
        assert loaded_sr == sr
        np.testing.assert_allclose(loaded, audio, atol=1e-6)

    def test_rejects_out_of_range_values(self, tmp_path):
        bad = np.array([0.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match=r"\[-1, 1\]"):
            save_audio(tmp_path / "bad.wav", bad, 16000)
