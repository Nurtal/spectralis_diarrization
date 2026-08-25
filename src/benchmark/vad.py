"""Simple energy-based voice activity detector (Phase 2 baseline module)."""

import numpy as np

from benchmark.interfaces import Segment

FRAME_MS = 30


def energy_vad(audio, sample_rate, threshold_db=-40.0, min_speech_s=0.1, frame_ms=FRAME_MS):
    """Return speech Segments based on frame RMS above `threshold_db` dBFS.

    Frames shorter than `min_speech_s` are discarded.
    """
    audio = np.asarray(audio, dtype=np.float32)
    frame_len = max(1, int(sample_rate * frame_ms / 1000))
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return []

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-10))
    voiced = db > threshold_db

    segments = []
    start_frame = None
    for i, v in enumerate(voiced):
        if v and start_frame is None:
            start_frame = i
        elif not v and start_frame is not None:
            _append_segment(segments, start_frame, i, sample_rate, frame_ms, min_speech_s)
            start_frame = None
    if start_frame is not None:
        _append_segment(segments, start_frame, n_frames, sample_rate, frame_ms, min_speech_s)
    return segments


def _append_segment(segments, start_frame, end_frame, sr, frame_ms, min_speech_s):
    start = start_frame * frame_ms / 1000.0
    end = end_frame * frame_ms / 1000.0
    if end - start >= min_speech_s:
        segments.append(Segment(start=start, end=end, speaker="speech"))
