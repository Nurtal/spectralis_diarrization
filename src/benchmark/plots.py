"""Plotting helpers. matplotlib is an optional dependency (group `viz`)."""

from pathlib import Path


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is not installed. Install it with `uv sync --group viz`."
        ) from e
    return plt


def _all_speakers(segments):
    return sorted({s.speaker for s in segments})


def _max_end(segments_a, segments_b):
    m = 0.0
    for s in list(segments_a) + list(segments_b):
        if s.end > m:
            m = s.end
    return m


def _overlap_intervals(segments):
    """Exact intervals where >=2 speakers overlap."""
    if not segments:
        return []
    bounds = sorted({p for s in segments for p in (s.start, s.end)})
    intervals = []
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2.0
        active = sum(1 for s in segments if s.start <= mid < s.end)
        if active >= 2:
            intervals.append((a, b))
    # merge contiguous
    merged = []
    for s, e in intervals:
        if merged and abs(merged[-1][1] - s) < 1e-9:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def plot_quality_vs_cost(rows, quality, cost, out):
    """Scatter of quality vs runtime per model."""
    plt = _pyplot()
    by_model = {}
    for r in rows:
        if isinstance(r.get(quality), (int, float)) and isinstance(r.get(cost), (int, float)):
            by_model.setdefault(r["model"], []).append((r[cost], r[quality]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, points in sorted(by_model.items()):
        xs, ys = zip(*points)
        ax.scatter(xs, ys, label=model)
    ax.set_xlabel(cost)
    ax.set_ylabel(quality)
    ax.legend()
    ax.set_title("Quality vs cost")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


def plot_metric_by_model(rows, metric, out):
    """Bar chart of mean metric per model."""
    plt = _pyplot()
    by_model = {}
    for r in rows:
        if isinstance(r.get(metric), (int, float)):
            by_model.setdefault(r["model"], []).append(r[metric])
    models = sorted(by_model)
    means = [sum(v) / len(v) for v in (by_model[m] for m in models)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, means)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by model")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


def plot_sweep(rows, x_key, y_key, out):
    """One curve per model over a swept variable (overlap, SNR, speakers...)."""
    plt = _pyplot()
    series = {}
    for r in rows:
        if isinstance(r.get(x_key), (int, float)) and isinstance(r.get(y_key), (int, float)):
            series.setdefault(r["model"], []).append((r[x_key], r[y_key]))
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, points in sorted(series.items()):
        pts = sorted(points)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=model)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.legend()
    ax.set_title(f"{y_key} vs {x_key}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return Path(out)


def plot_diarization_timeline(reference, hypothesis, out, title=None):
    """GT vs hypothesis timeline, one lane per speaker, overlap hatched."""
    plt = _pyplot()
    import matplotlib.patches as mpatches

    reference = list(reference or [])
    hypothesis = list(hypothesis or [])

    speakers = sorted(set(_all_speakers(reference) + _all_speakers(hypothesis)))
    # fallback for VAD single label "speech"
    if not speakers:
        speakers = []

    max_end = _max_end(reference, hypothesis)
    if max_end <= 0:
        max_end = 1.0

    # color per speaker, deterministic via tab10
    cmap = plt.get_cmap("tab10")
    color_map = {spk: cmap(i % 10) for i, spk in enumerate(speakers)}

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10, max(3, 1.2 * max(2, len(speakers)) + 1)),
        gridspec_kw={"hspace": 0.4},
        layout="constrained",
    )
    ax_ref, ax_hyp = axes

    def _draw(ax, segments, label):
        # draw per speaker lane
        for idx, spk in enumerate(speakers):
            y = idx * 10
            segs = [(s.start, s.end - s.start) for s in segments if s.speaker == spk]
            if segs:
                ax.broken_barh(
                    segs, (y, 8), facecolors=color_map[spk], edgecolors="black", linewidth=0.5
                )
        # overlap background
        for s, e in _overlap_intervals(segments):
            ax.axvspan(s, e, color="red", alpha=0.12, hatch="///", zorder=0)
        ax.set_ylim(-2, len(speakers) * 10 + 2 if speakers else 10)
        ax.set_yticks([i * 10 + 4 for i in range(len(speakers))] if speakers else [])
        ax.set_yticklabels(speakers if speakers else [])
        ax.set_title(label, fontsize=10, loc="left")
        ax.grid(axis="x", alpha=0.3)
        if not segments:
            ax.text(
                max_end / 2,
                (len(speakers) * 10) / 2 if speakers else 5,
                "no speech",
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
                style="italic",
            )

    _draw(ax_ref, reference, "Reference (GT)")
    _draw(ax_hyp, hypothesis, "Hypothesis")

    for ax in axes:
        ax.set_xlim(0, max_end)
        ax.set_xlabel("time (s)")

    if title:
        fig.suptitle(title, fontsize=11)

    # legend for speakers
    if speakers:
        patches = [mpatches.Patch(color=color_map[s], label=s) for s in speakers]
        # overlap patch
        patches.append(mpatches.Patch(facecolor="red", alpha=0.12, hatch="///", label="overlap"))
        fig.legend(handles=patches, loc="outside upper right", fontsize=7, ncol=1)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_overlap_heatmap(reference, hypothesis, out, title=None):
    """Binary overlap heatmap: GT vs HYP overlap over time (10 ms grid)."""
    plt = _pyplot()
    import numpy as np

    reference = list(reference or [])
    hypothesis = list(hypothesis or [])
    max_end = _max_end(reference, hypothesis)
    if max_end <= 0:
        max_end = 1.0

    # 10 ms grid matching metrics.GRID_STEP_S
    step = 0.01
    grid = np.arange(0.0, max_end, step)
    if len(grid) == 0:
        grid = np.array([0.0])

    def _mask(segments):
        if not segments:
            return np.zeros(len(grid), dtype=int)
        # count active speakers per frame
        counts = np.zeros(len(grid), dtype=int)
        for s in segments:
            counts[(grid >= s.start) & (grid < s.end)] += 1
        return (counts >= 2).astype(int)

    ref_ov = _mask(reference)
    hyp_ov = _mask(hypothesis)
    data = np.vstack([ref_ov, hyp_ov])

    # compute overlap detection scores for annotation
    both = int(((ref_ov == 1) & (hyp_ov == 1)).sum())
    prec = both / int((hyp_ov == 1).sum()) if (hyp_ov == 1).any() else 0.0
    rec = both / int((ref_ov == 1).sum()) if (ref_ov == 1).any() else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    fig, ax = plt.subplots(figsize=(10, 2.5))
    im = ax.imshow(data, aspect="auto", cmap="Reds", vmin=0, vmax=1, extent=[0, max_end, -0.5, 1.5])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GT overlap", "HYP overlap"])
    ax.set_xlabel("time (s)")
    ax.set_title(title or f"Overlap detection — P {prec:.2f} R {rec:.2f} F1 {f1:.2f}")
    # colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1], shrink=0.6)
    cbar.ax.set_yticklabels(["no", "yes"])
    # grid lines
    ax.set_xlim(0, max_end)
    ax.grid(axis="x", alpha=0.2)

    if max_end > 20:
        # too long: annotate that we downsample visually via extent, not actual grid
        ax.text(
            0.99,
            0.02,
            f"grid {step * 1000:.0f} ms",
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=6,
            color="gray",
        )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_der_breakdown(metrics_list, out, title=None):
    """Stacked bar: missed / false_alarm / confusion per mixture."""
    plt = _pyplot()

    metrics_list = list(metrics_list or [])
    fig, ax = plt.subplots(figsize=(max(6, len(metrics_list) * 0.8), 4))

    if not metrics_list:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_axis_off()
        ax.set_title(title or "DER breakdown (empty)")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        return out

    x = range(len(metrics_list))
    missed = [float(m.get("missed", 0.0)) for m in metrics_list]
    fa = [float(m.get("false_alarm", 0.0)) for m in metrics_list]
    conf = [float(m.get("confusion", 0.0)) for m in metrics_list]

    ax.bar(x, missed, label="missed", color="#d62728")
    ax.bar(x, fa, bottom=missed, label="false alarm", color="#ff7f0e")
    bottom2 = [a + b for a, b in zip(missed, fa)]
    ax.bar(x, conf, bottom=bottom2, label="confusion", color="#1f77b4")

    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [m.get("label", f"mix {i}") for i, m in enumerate(metrics_list)],
        rotation=30,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("seconds")
    ax.set_xlabel("mixture")
    ax.set_title(title or "DER breakdown per mixture (FA / missed / confusion)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
