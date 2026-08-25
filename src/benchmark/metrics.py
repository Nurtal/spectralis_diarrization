"""Diarization metrics: DER (with decomposition), overlap detection.

Scoring is done on a fixed time grid (10 ms by default), which makes the
computation deterministic and simple while staying accurate far below the
resolution of any diarizer output.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

GRID_STEP_S = 0.01


def _active_arrays(segments, grid):
    end = float(grid[-1] + GRID_STEP_S)
    speakers = sorted({s.speaker for s in segments})
    active = {}
    for spk in speakers:
        arr = np.zeros(len(grid), dtype=bool)
        for seg in segments:
            if seg.speaker == spk:
                arr[(grid >= seg.start) & (grid < min(seg.end, end))] = True
        active[spk] = arr
    return speakers, active


def _grid_for(segments_a, segments_b):
    max_end = 0.0
    for seg in list(segments_a) + list(segments_b):
        max_end = max(max_end, seg.end)
    if max_end <= 0:
        return np.zeros(0)
    return np.arange(0.0, max_end, GRID_STEP_S)


def _map_hypothesis_to_reference(ref_active, hyp_speakers, hyp_active):
    """Optimal injective assignment hyp->ref maximizing co-occurrence.

    Returns a dict hyp_speaker -> ref_speaker or None when unmapped.
    """
    ref_speakers = list(ref_active)
    n_ref, n_hyp = len(ref_speakers), len(hyp_speakers)
    size = max(n_ref, n_hyp)
    contingency = np.zeros((size, size))
    for i, rs in enumerate(ref_speakers):
        for j, hs in enumerate(hyp_speakers):
            contingency[i, j] = int(np.sum(ref_active[rs] & hyp_active[hs]))
    row_ind, col_ind = linear_sum_assignment(-contingency)
    mapping = {hs: None for hs in hyp_speakers}
    for i, j in zip(row_ind, col_ind):
        if i < n_ref and j < n_hyp:
            mapping[hyp_speakers[j]] = ref_speakers[i]
    return mapping


def der(reference, hypothesis, step=None):
    """Diarization Error Rate with FA / missed / confusion decomposition.

    Times are in seconds of reference speech. Returns a dict with keys
    `der`, `total`, `false_alarm`, `missed`, `confusion` (all in seconds,
    except `der` which is a fraction).
    """
    step = step or GRID_STEP_S
    grid = _grid_for(reference, hypothesis)
    result = {
        "der": 0.0,
        "total": 0.0,
        "false_alarm": 0.0,
        "missed": 0.0,
        "confusion": 0.0,
    }
    if len(grid) == 0:
        return result

    _, ref_active = _active_arrays(reference, grid)
    ref_speakers = list(ref_active)
    hyp_speakers, hyp_active = _active_arrays(hypothesis, grid)
    mapping = _map_hypothesis_to_reference(ref_active, hyp_speakers, hyp_active)

    n_ref = len(ref_speakers)
    ref_matrix = (
        np.stack([ref_active[s] for s in ref_speakers])
        if n_ref
        else np.zeros((0, len(grid)), dtype=bool)
    )

    total = float(ref_matrix.sum()) * step
    correct_pts = np.zeros(len(grid), dtype=np.int64)

    for hs in hyp_speakers:
        if mapping.get(hs) is not None:
            i = ref_speakers.index(mapping[hs])
            covered = hyp_active[hs] & ref_matrix[i]
            correct_pts += covered.astype(np.int64)

    n_ref_pts = ref_matrix.sum(axis=0)
    n_hyp_pts = sum(hyp_active[hs].astype(np.int64) for hs in hyp_speakers) if hyp_speakers else 0

    confusion_pts = np.minimum(n_ref_pts, n_hyp_pts) - correct_pts
    missed_pts = np.maximum(n_ref_pts - n_hyp_pts, 0)
    fa_pts = np.maximum(n_hyp_pts - n_ref_pts, 0)

    false_alarm = float(fa_pts.sum()) * step
    missed = float(missed_pts.sum()) * step
    confusion = float(confusion_pts.sum()) * step

    result.update(
        total=total,
        false_alarm=false_alarm,
        missed=missed,
        confusion=confusion,
        der=(missed + false_alarm + confusion) / total if total > 0 else 0.0,
    )
    return result


def jer(reference, hypothesis, step=None):
    """Jaccard Error Rate: mean over reference speakers of 1 - J(i)."""
    step = step or GRID_STEP_S
    grid = _grid_for(reference, hypothesis)
    if len(grid) == 0 or not reference:
        return 0.0
    ref_speakers, ref_active = _active_arrays(reference, grid)
    hyp_speakers, hyp_active = _active_arrays(hypothesis, grid)
    mapping = _map_hypothesis_to_reference(ref_active, hyp_speakers, hyp_active)

    errors = []
    for rs in ref_speakers:
        mapped_hs = [hs for hs in hyp_speakers if mapping.get(hs) == rs]
        hyp_union = np.zeros(len(grid), dtype=bool)
        for hs in mapped_hs:
            hyp_union |= hyp_active[hs]
        union = ref_active[rs] | hyp_union
        inter = ref_active[rs] & hyp_union
        u = int(union.sum())
        errors.append(1.0 - (int(inter.sum()) / u) if u > 0 else 1.0)
    return float(np.mean(errors))


def overlap_detection_scores(reference, hypothesis, step=None):
    """Precision/recall/F1 of detecting time spans where >=2 speakers talk."""
    step = step or GRID_STEP_S
    grid = _grid_for(reference, hypothesis)
    scores = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if len(grid) == 0:
        return scores

    def overlap_mask(segments):
        _, active = _active_arrays(segments, grid)
        if not active:
            return np.zeros(len(grid), dtype=bool)
        return sum(a.astype(np.int64) for a in active.values()) >= 2

    ref_ovr = overlap_mask(reference)
    hyp_ovr = overlap_mask(hypothesis)
    both = float((ref_ovr & hyp_ovr).sum())
    precision = both / float(hyp_ovr.sum()) if hyp_ovr.any() else 0.0
    recall = both / float(ref_ovr.sum()) if ref_ovr.any() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    scores.update(precision=precision, recall=recall, f1=f1)
    return scores
