"""Enrollment policy: pick clean solo speech per speaker from diarization output."""

import numpy as np


def solo_spans(segments, speaker):
    """Merged time spans where only `speaker` is active."""
    own = [(s.start, s.end) for s in segments if s.speaker == speaker]
    if not own:
        return []
    others = [(s.start, s.end) for s in segments if s.speaker != speaker]

    def subtract(span, cuts):
        s, e = span
        pieces = [(s, e)]
        for c0, c1 in cuts:
            nxt = []
            for p0, p1 in pieces:
                if c1 <= p0 or c0 >= p1:
                    nxt.append((p0, p1))
                    continue
                if p0 < c0:
                    nxt.append((p0, min(c0, p1)))
                if c1 < p1:
                    nxt.append((max(c1, p0), p1))
            pieces = [(a, b) for a, b in nxt if b > a]
        return pieces

    spans = []
    for span in own:
        spans.extend(subtract(span, others))

    # merge adjacent/nearby pieces
    spans.sort()
    merged = []
    for s, e in spans:
        if merged and s - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def select_enrollment_segments(segments, speaker, max_duration=10.0):
    """Longest-first selection of clean solo spans capped at `max_duration`.

    Returns spans in chronological order.
    """
    spans = sorted(solo_spans(segments, speaker), key=lambda s: s[1] - s[0], reverse=True)
    chosen, total = [], 0.0
    for s, e in spans:
        remaining = max_duration - total
        if remaining <= 0:
            break
        take = min(e - s, remaining)
        chosen.append((s, s + take))
        total += take
    return sorted(chosen)


def enrollment_audio(audio, sample_rate, spans):
    """Concatenate `audio` over the given (start, end) second spans."""
    parts = []
    for s, e in spans:
        i0, i1 = int(s * sample_rate), int(e * sample_rate)
        parts.append(np.asarray(audio)[i0:i1])
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


def degraded_enrollment(audio, sample_rate, spans, snr_db, seed=0):
    """Enrollment audio with additive noise at `snr_db` (robustness studies)."""
    clip = enrollment_audio(audio, sample_rate, spans)
    if snr_db is None:
        return clip
    from benchmark.generator import mix_with_noise

    noisy, _ = mix_with_noise(clip, snr_db=snr_db, seed=seed)
    return noisy
