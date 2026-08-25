"""Robustness studies for speaker-conditioned pipelines (Phase 6, RQ5)."""

import numpy as np

from benchmark.enrollment import degraded_enrollment, enrollment_audio


def embedding_robustness_curve(
    encoder, audio, sample_rate, spans, snrs, speaker="enrollment", seed=0
):
    """Cosine similarity between clean and degraded enrollment embeddings.

    Returns one record per SNR level: {"speaker", "snr", "cosine"}. `snr=None`
    means clean enrollment (similarity exactly 1.0 by construction).
    """
    clean_clip = enrollment_audio(audio, sample_rate, spans)
    if len(clean_clip) == 0:
        raise ValueError("no enrollment audio for the given spans")
    clean = encoder.encode(clean_clip, sample_rate)

    records = []
    for snr in snrs:
        if snr is None:
            records.append({"speaker": speaker, "snr": None, "cosine": 1.0})
            continue
        noisy = degraded_enrollment(audio, sample_rate, spans, snr_db=snr, seed=seed)
        emb = encoder.encode(noisy, sample_rate)
        records.append({"speaker": speaker, "snr": float(snr), "cosine": float(np.dot(clean, emb))})
    return records
