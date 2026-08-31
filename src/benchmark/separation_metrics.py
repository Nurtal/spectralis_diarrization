"""Source separation metrics: SI-SDR, SDR/SIR/SAR, PESQ/STOI.

The BSS metrics use the bss_eval decomposition without distortion filters:
estimate = s_target + e_interference + e_artifacts, where projections are
plain least-squares onto the reference subspace.

PESQ/STOI are optional speech-quality metrics (wideband 16 kHz only for PESQ,
per ADR-007). They require ``pesq`` and ``pystoi`` (``uv sync --group quality``)
and degrade gracefully when the libraries are missing.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _zero_mean(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean()


def si_sdr(reference, estimate, eps=1e-8):
    """Scale-invariant SDR in dB between one reference and one estimate."""
    ref = _zero_mean(reference)
    est = _zero_mean(estimate)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    target = alpha * ref
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps)))


def best_pairing_si_sdr(references, estimates):
    """Mean SI-SDR after optimal permutation of estimates onto references."""
    n = min(len(references), len(estimates))
    if n == 0:
        return float("-inf")
    cost = np.array([[-si_sdr(r, e) for e in estimates] for r in references])
    row, col = linear_sum_assignment(cost[:, : len(estimates)][:n, :][:, :n])
    scores = [-cost[i, j] for i, j in zip(row, col)]
    return float(np.mean(scores))


def _project(x, basis):
    """Least-squares projection of x onto span(basis columns)."""
    if basis.size == 0 or not basis.any():
        return np.zeros_like(x)
    coef, *_ = np.linalg.lstsq(basis.T, x, rcond=None)
    return basis.T @ coef


def _pair_metrics(reference, estimate):
    ref = _zero_mean(reference)
    est = _zero_mean(estimate)
    eps = 1e-8

    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    s_target = alpha * ref

    return s_target, est - s_target


def bss_metrics(references, estimates, pairing=None):
    """Simplified SDR/SIR/SAR (dB), averaged over optimally paired sources.

    `pairing` is optional: a list mapping estimate index -> reference index.
    Computed by SI-SDR when omitted.
    """
    n = min(len(references), len(estimates))
    if n == 0:
        return {"sdr": 0.0, "sir": 0.0, "sar": 0.0}

    if pairing is None:
        cost = np.array([[si_sdr(r, e) for e in estimates] for r in references])
        row, col = linear_sum_assignment(-cost)
        pairing = {int(j): int(i) for i, j in zip(row, col)}

    refs_zm = [_zero_mean(r) for r in references]
    ests_zm = [_zero_mean(e) for e in estimates]

    sdr_values, sir_values, sar_values = [], [], []
    for est_idx in sorted(pairing)[:n]:
        ref_idx = pairing[est_idx]
        est = ests_zm[est_idx]
        ref = refs_zm[ref_idx]

        alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
        s_target = alpha * ref
        residual = est - s_target

        others = [refs_zm[k] for k in range(len(refs_zm)) if k != ref_idx]
        if others:
            e_interf = _project(residual, np.stack(others))
        else:
            e_interf = np.zeros_like(residual)
        e_artif = residual - e_interf

        eps = 1e-8
        p_target = np.dot(s_target, s_target)
        p_interf = np.dot(e_interf, e_interf)
        p_artif = np.dot(e_artif, e_artif)

        sdr_values.append(10 * np.log10((p_target + eps) / (p_interf + p_artif + eps)))
        sir_values.append(10 * np.log10((p_target + eps) / (p_interf + eps)))
        sar_values.append(10 * np.log10((p_target + p_interf + eps) / (p_artif + eps)))

    return {
        "sdr": float(np.mean(sdr_values)),
        "sir": float(np.mean(sir_values)),
        "sar": float(np.mean(sar_values)),
    }


def _pesq_available():
    try:
        import pesq  # noqa: F401

        return True
    except ImportError:
        return False


def _stoi_available():
    try:
        import pystoi  # noqa: F401

        return True
    except ImportError:
        return False


def pesq_score(reference, estimate, sample_rate=16000):
    """PESQ wideband (MOS 1.0–4.5) for a single pair. Requires 16 kHz.

    Raises ``RuntimeError`` when the ``pesq`` package is missing and
    ``ValueError`` for unsupported sample rates.
    """
    if int(sample_rate) != 16000:
        raise ValueError("pesq_score only supports sample_rate=16000 (wideband) for T2")
    if not _pesq_available():
        raise RuntimeError(
            "pesq is not installed. Install it with `uv sync --group quality` "
            "(or `pip install pesq`)."
        )
    from pesq import pesq as pesq_fn

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.ndim != 1 or est.ndim != 1:
        raise ValueError("reference and estimate must be 1D waveforms")
    n = min(len(ref), len(est))
    if n == 0:
        raise ValueError("reference and estimate must be non-empty")
    ref = ref[:n]
    est = est[:n]
    # PESQ requires at least ~0.25 s; let the library raise a clear error
    # if the signal is too short, but provide a nicer message for empty
    return float(pesq_fn(int(sample_rate), ref, est, "wb"))


def stoi_score(reference, estimate, sample_rate=16000):
    """STOI (0–1) for a single pair. Supports any rate via pystoi, but 16 kHz is the T2 default."""
    if not _stoi_available():
        raise RuntimeError(
            "pystoi is not installed. Install it with `uv sync --group quality` "
            "(or `pip install pystoi`)."
        )
    from pystoi import stoi as stoi_fn

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.ndim != 1 or est.ndim != 1:
        raise ValueError("reference and estimate must be 1D waveforms")
    n = min(len(ref), len(est))
    if n == 0:
        raise ValueError("reference and estimate must be non-empty")
    ref = ref[:n]
    est = est[:n]
    return float(stoi_fn(ref, est, int(sample_rate), extended=False))


def speech_quality_metrics(references, estimates, sample_rate=16000, pairing=None):
    """Mean PESQ/STOI over optimally paired sources (permutation-invariant).

    Returns ``{}`` when neither ``pesq`` nor ``pystoi`` is installed, or
    ``{\"pesq\": ..., \"stoi\": ...}`` with only the available keys.
    Pairing is derived from SI-SDR when not supplied, matching ``bss_metrics``.
    """
    if not _pesq_available() and not _stoi_available():
        return {}
    n = min(len(references), len(estimates))
    if n == 0:
        return {}

    if pairing is None:
        cost = np.array([[si_sdr(r, e) for e in estimates] for r in references])
        row, col = linear_sum_assignment(-cost)
        pairing = {int(j): int(i) for i, j in zip(row, col)}

    pesq_vals, stoi_vals = [], []
    for est_idx in sorted(pairing)[:n]:
        ref_idx = pairing[est_idx]
        ref = references[ref_idx]
        est = estimates[est_idx]
        if _pesq_available():
            try:
                pesq_vals.append(pesq_score(ref, est, sample_rate))
            except Exception:
                # e.g. too short, unsupported rate — skip this pair
                pass
        if _stoi_available():
            try:
                stoi_vals.append(stoi_score(ref, est, sample_rate))
            except Exception:
                pass

    result = {}
    if pesq_vals:
        result["pesq"] = float(np.mean(pesq_vals))
    if stoi_vals:
        result["stoi"] = float(np.mean(stoi_vals))
    return result
