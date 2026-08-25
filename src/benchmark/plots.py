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
