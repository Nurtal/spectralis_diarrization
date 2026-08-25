from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_audio(path, sample_rate=None):
    """Load an audio file as float32.

    Returns (audio, sr) where audio has shape (samples,) for mono files or
    (samples, channels) otherwise, with values in [-1, 1]. If `sample_rate`
    is given and differs from the file's rate, the audio is resampled.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate is not None and sample_rate != sr:
        g = gcd(int(sample_rate), int(sr))
        up = int(sample_rate) // g
        down = sr // g
        audio = resample_poly(audio, up, down, axis=0).astype(np.float32)
        sr = int(sample_rate)
    if audio.shape[1] == 1:
        audio = audio[:, 0]
    return np.ascontiguousarray(audio), sr


def save_audio(path, audio, sample_rate):
    """Write float32 audio (mono or (samples, channels)) to a file.

    Wav files default to the lossless FLOAT subtype. Raises ValueError if
    values fall outside [-1, 1].
    """
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size and (audio.max() > 1.0 or audio.min() < -1.0):
        raise ValueError("Audio values must be in [-1, 1]")
    sf.write(str(path), audio, sample_rate, subtype="FLOAT")
