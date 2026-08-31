import numpy as np
import pytest

from benchmark.asr_metrics import cer_score, speech_recognition_metrics, transcribe, wer_score


def test_wer_perfect_is_zero():
    assert wer_score("hello world", "hello world") == pytest.approx(0.0)


def test_wer_one_error():
    # 1 substitution out of 2 words = 0.5
    assert wer_score("hello world", "hello earth") == pytest.approx(0.5)


def test_cer_perfect():
    assert cer_score("hello", "hello") == pytest.approx(0.0)


def test_transcribe_fallback_without_libs(monkeypatch):
    import benchmark.asr_metrics as mod

    monkeypatch.setattr(mod, "_asr_available", lambda: False)
    with pytest.raises(RuntimeError, match="whisper"):
        transcribe(np.zeros(8000, dtype=np.float32), 16000)


def test_speech_recognition_fallback_without_libs(monkeypatch):
    import benchmark.asr_metrics as mod

    monkeypatch.setattr(mod, "_asr_available", lambda: False)
    monkeypatch.setattr(mod, "_wer_available", lambda: False)
    ref = np.zeros(8000, dtype=np.float32)
    est = np.zeros(8000, dtype=np.float32)
    assert speech_recognition_metrics([ref], [est], 16000) == {}


def test_speech_recognition_with_fake_asr(monkeypatch):
    import benchmark.asr_metrics as mod

    # fake transcribe returns fixed text based on audio content
    def fake_transcribe(audio, sr, model="tiny"):
        # simple: if audio is all zeros -> "hello world", else "hello earth"
        if np.abs(audio).max() < 1e-6:
            return "hello world"
        return "hello earth"

    monkeypatch.setattr(mod, "transcribe", fake_transcribe)
    monkeypatch.setattr(mod, "_asr_available", lambda: True)
    monkeypatch.setattr(mod, "_wer_available", lambda: True)

    ref = np.zeros(8000, dtype=np.float32)
    est = np.ones(8000, dtype=np.float32) * 0.1
    m = speech_recognition_metrics([ref], [est], 16000)
    # ref -> hello world, est -> hello earth => WER 0.5
    assert m["wer"] == pytest.approx(0.5)
    assert "cer" in m


def test_wer_case_insensitive():
    assert wer_score("Hello World", "hello world") == pytest.approx(0.0)
