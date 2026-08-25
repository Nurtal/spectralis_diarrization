"""Source separation metrics: SI-SDR and simplified SDR/SIR/SAR.

The BSS metrics use the bss_eval decomposition without distortion filters:
estimate = s_target + e_interference + e_artifacts, where projections are
plain least-squares onto the reference subspace.
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
