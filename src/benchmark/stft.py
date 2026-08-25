"""STFT/iSTFT utilities shared by TF-domain separation approaches."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import istft as scipy_istft
from scipy.signal import stft as scipy_stft


@dataclass(frozen=True)
class StftParams:
    sample_rate: int
    n_fft: int
    n_overlap: int


DEFAULT_N_FFT = 512
DEFAULT_OVERLAP_RATIO = 0.75


def stft(audio, sample_rate, n_fft=DEFAULT_N_FFT, overlap=DEFAULT_OVERLAP_RATIO):
    """Complex STFT of a mono signal. Returns (spectrogram [freq, time], params)."""
    audio = np.asarray(audio, dtype=np.float32)
    n_fft = min(int(n_fft), max(len(audio), 2))
    n_overlap = min(int(n_fft * overlap), n_fft - 1)
    _, _, Z = scipy_stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=n_fft,
        noverlap=n_overlap,
        padded=True,
    )
    return Z.astype(np.complex64), StftParams(sample_rate, n_fft, n_overlap)


def istft(spectrogram, params, length=None):
    """Inverse STFT. Returns (waveform, sample_rate)."""
    times, reconstructed = scipy_istft(
        spectrogram,
        fs=params.sample_rate,
        window="hann",
        nperseg=params.n_fft,
        noverlap=params.n_overlap,
    )
    if length is not None:
        if len(reconstructed) < length:
            reconstructed = np.pad(reconstructed, (0, length - len(reconstructed)))
        else:
            reconstructed = reconstructed[:length]
    return reconstructed.astype(np.float32), params.sample_rate
