"""Blind NMF speech separation: the classical Phase 3 baseline.

The mixture magnitude spectrogram is factorized as V ~= W @ H with
multiplicative updates. Components are clustered into `num_speakers` groups
by spectral similarity (k-means on L2-normalized spectral profiles), and
per-cluster soft Wiener masks reconstruct each source.

Known limitation (quantified in tests): simultaneous *stationary* sources in
overlapping bands are a degenerate case for blind NMF. Separation relies on
sources differing in spectral or temporal structure, as real speakers do.
"""

import numpy as np

from benchmark.interfaces import Separator
from benchmark.stft import istft, stft

DEFAULT_COMPONENTS_PER_SOURCE = 8
DEFAULT_ITERS = 100


def _kmeans(points, k, rng, iters=50, n_init=8):
    """Lloyd's algorithm with k-means++ init and best-of-restarts selection."""
    best_centers, best_inertia = None, np.inf
    for _ in range(n_init):
        centers = _kmeans_pp_init(points, k, rng)
        for _ in range(iters):
            distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            new_centers = centers.copy()
            for c in range(k):
                mask = labels == c
                if mask.any():
                    new_centers[c] = points[mask].mean(axis=0)
            if np.allclose(new_centers, centers):
                break
            centers = new_centers
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia = distances.min(axis=1).sum()
        if inertia < best_inertia:
            best_inertia, best_centers = inertia, centers
    distances = ((points[:, None, :] - best_centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1), best_centers


def _kmeans_pp_init(points, k, rng):
    """k-means++ seeding: spread initial centers apart."""
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        distances = np.array([min(((p - c) ** 2).sum() for c in centers) for p in points])
        total = distances.sum()
        if total == 0:
            centers.append(points[rng.integers(len(points))])
        else:
            centers.append(points[rng.choice(len(points), p=distances / total)])
    return np.array(centers)


class NMFSeparator(Separator):
    def __init__(
        self,
        num_speakers=None,
        components_per_source=DEFAULT_COMPONENTS_PER_SOURCE,
        iterations=DEFAULT_ITERS,
        seed=0,
    ):
        self.num_speakers = num_speakers
        self.components_per_source = components_per_source
        self.iterations = iterations
        self.seed = seed

    def separate(self, audio, sample_rate, num_speakers=None):
        k = num_speakers or self.num_speakers
        if not k or k < 1:
            raise ValueError("num_speakers must be a positive integer for NMF")

        audio = self._validate_audio(audio, sample_rate)
        spec, params = stft(audio, sample_rate)
        magnitude = np.maximum(np.abs(spec), 1e-10)
        n_freq, n_time = magnitude.shape

        n_components = min(k * self.components_per_source, min(n_freq, n_time))
        w, h = self._nmf(magnitude, n_components)

        # cluster components on their L2-normalized spectral profile,
        # discarding near-dead components first
        eps = 1e-10
        rng = np.random.default_rng(self.seed)
        energy = np.einsum("fc,ct->c", w, h)
        alive = energy > max(energy.max() * 1e-2, 1e-8)
        features = w[:, alive].T.astype(np.float64)  # [components, freq]
        features /= np.linalg.norm(features, axis=1, keepdims=True) + eps
        labels_alive, _ = _kmeans(features, k, rng)
        labels = np.full(n_components, -1, dtype=np.int64)
        labels[alive] = labels_alive

        denom = (w @ h).sum(axis=0) + eps
        masks = []
        for c in range(k):
            comp_mask = np.isin(labels, [c])
            source_power = (w[:, comp_mask] @ h[comp_mask]).sum(axis=0)
            masks.append(source_power / denom)

        sources = []
        for mask in masks:
            masked = spec * mask[None, :]
            waveform, _ = istft(masked, params, length=len(audio))
            sources.append(waveform)
        return sources

    def _nmf(self, v, n_components, eps=1e-10):
        """Euclidean multiplicative updates for V ~= W @ H.

        Components are initialized as spectral spikes at distinct random
        frequency bins so the factorization covers the whole spectrogram
        instead of several components collapsing onto the loudest peak.
        """
        rng = np.random.default_rng(self.seed)
        n_freq, n_time = v.shape
        bins = rng.choice(n_freq, size=n_components, replace=min(n_components, n_freq) > n_freq)
        w = np.full((n_freq, n_components), eps)
        w[bins, np.arange(n_components)] = v.max() * 0.5
        h = rng.uniform(0.1, 1.0, size=(n_components, n_time))
        for _ in range(self.iterations):
            w *= (v @ h.T) / (w @ h @ h.T + eps)
            h *= (w.T @ v) / (w.T @ w @ h + eps)
        return w, h
