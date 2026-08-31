"""ASR-based metrics: WER/CER via Whisper + jiwer (RQ9, ADR-007).

Whisper and jiwer are optional (``uv sync --group asr``). When missing,
functions raise ``RuntimeError`` with an install hint, and the aggregated
``speech_recognition_metrics`` degrades gracefully to ``{}`` so evaluation
never fails hard.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from benchmark.separation_metrics import si_sdr


def _asr_available():
    try:
        import whisper  # noqa: F401

        return True
    except ImportError:
        return False


def _wer_available():
    try:
        import jiwer  # noqa: F401

        return True
    except ImportError:
        return False


def _normalize_text(text):
    return " ".join(str(text).strip().lower().split())


def wer_score(reference_text, hypothesis_text):
    """Word Error Rate (0.0–inf, lower is better)."""
    if not _wer_available():
        raise RuntimeError("jiwer not installed. Run `uv sync --group asr`.")
    import jiwer

    ref = _normalize_text(reference_text)
    hyp = _normalize_text(hypothesis_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def cer_score(reference_text, hypothesis_text):
    """Character Error Rate."""
    if not _wer_available():
        raise RuntimeError("jiwer not installed. Run `uv sync --group asr`.")
    import jiwer

    ref = _normalize_text(reference_text)
    hyp = _normalize_text(hypothesis_text)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.cer(ref, hyp))


# simple in-process cache for whisper models (tiny is ~150 MB, load once)
_WHISPER_CACHE = {}


def _load_whisper(model_name="tiny"):
    if not _asr_available():
        raise RuntimeError("whisper not installed. Run `uv sync --group asr`.")
    if model_name in _WHISPER_CACHE:
        return _WHISPER_CACHE[model_name]
    import whisper

    model = whisper.load_model(model_name)
    _WHISPER_CACHE[model_name] = model
    return model


def transcribe(audio, sample_rate, model="tiny"):
    """Transcribe a mono waveform with Whisper (16 kHz expected)."""
    if not _asr_available():
        raise RuntimeError("whisper not installed. Run `uv sync --group asr`.")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 1:
        raise ValueError("audio must be 1D mono")
    if len(audio) == 0:
        return ""
    # Whisper expects 16 kHz; resample if needed
    if int(sample_rate) != 16000:
        from benchmark.neural_separator import _resample

        audio = _resample(audio, int(sample_rate), 16000)
    # whisper.transcribe expects a file path; use a temp wav
    import tempfile

    from benchmark.audio import save_audio

    model_obj = _load_whisper(model)
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        save_audio(tmp.name, audio, 16000)
        result = model_obj.transcribe(tmp.name, language="en")
    return str(result.get("text", "")).strip()


def speech_recognition_metrics(
    references, estimates, sample_rate=16000, asr_model="tiny", pairing=None
):
    """Mean WER/CER over optimally paired sources via Whisper.

    Each reference is transcribed with Whisper to obtain a pseudo ground
    truth, then the paired estimate is transcribed and compared. This
    isolates the ASR degradation due to separation quality (RQ9).

    Returns ``{}`` when Whisper/jiwer are missing, or
    ``{"wer": ..., "cer": ..., "asr_model": ...}`` with available keys.
    """
    if not _asr_available() or not _wer_available():
        return {}
    n = min(len(references), len(estimates))
    if n == 0:
        return {}

    if pairing is None:
        cost = np.array([[si_sdr(r, e) for e in estimates] for r in references])
        row, col = linear_sum_assignment(-cost)
        pairing = {int(j): int(i) for i, j in zip(row, col)}

    wers, cers = [], []
    for est_idx in sorted(pairing)[:n]:
        ref_idx = pairing[est_idx]
        ref_audio = references[ref_idx]
        est_audio = estimates[est_idx]
        try:
            ref_text = transcribe(ref_audio, int(sample_rate), model=asr_model)
            hyp_text = transcribe(est_audio, int(sample_rate), model=asr_model)
        except Exception:
            continue
        try:
            wers.append(wer_score(ref_text, hyp_text))
            cers.append(cer_score(ref_text, hyp_text))
        except Exception:
            continue

    result = {}
    if wers:
        result["wer"] = float(np.mean(wers))
    if cers:
        result["cer"] = float(np.mean(cers))
    if result:
        result["asr_model"] = asr_model
    return result
