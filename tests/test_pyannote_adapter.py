import importlib.util

import numpy as np
import pytest

from benchmark.registry import DIARIZERS


def pyannote_available():
    try:
        return importlib.util.find_spec("pyannote") is not None
    except (ImportError, ValueError):
        return False


class TestPyannoteDiarizer:
    def test_registered(self):
        assert "pyannote" in DIARIZERS

    @pytest.mark.skipif(pyannote_available(), reason="pyannote installed")
    def test_helpful_error_when_missing(self):
        PyannoteDiarizer = DIARIZERS["pyannote"]
        with pytest.raises(RuntimeError, match="pip install"):
            PyannoteDiarizer()

    @pytest.mark.skipif(not pyannote_available(), reason="pyannote not installed")
    def test_runs_or_fails_with_informative_error(self):
        PyannoteDiarizer = DIARIZERS["pyannote"]
        try:
            d = PyannoteDiarizer()
        except RuntimeError as e:
            assert "token" in str(e).lower() or "hf auth login" in str(e)
            return
        segments = d.diarize(np.zeros(16000, dtype=np.float32), 16000)
        assert isinstance(segments, list)


class TestEnergyVadDiarizer:
    def test_registered_and_wraps_vad(self, tmp_path):
        from benchmark.audio import load_audio, save_audio
        from benchmark.vad import energy_vad

        assert "vad" in DIARIZERS
        sr = 8000
        wav = tmp_path / "a.wav"
        audio = np.concatenate(
            [
                np.zeros(sr, dtype=np.float32),
                0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32),
            ]
        )
        save_audio(wav, audio, sr)
        loaded, loaded_sr = load_audio(wav)

        d = DIARIZERS["vad"]()
        segments = d.diarize(loaded, loaded_sr)
        assert segments == energy_vad(loaded, loaded_sr)
        assert all(s.speaker == "speech" for s in segments)
