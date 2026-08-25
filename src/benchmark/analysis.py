"""Benchmark analysis: aggregation, tables, Pareto frontier, reports (Phase 7)."""

from pathlib import Path

from benchmark.results import read_results

TABLE_METRICS = ("si_sdr", "sdr", "sir", "sar", "der", "jer")


def aggregate(results_dir):
    """Flatten result JSONs into rows: model/experiment fields + metric values."""
    rows = []
    for record in read_results(results_dir):
        row = {k: v for k, v in record.items() if k not in ("metrics", "model_config", "dataset")}
        row.update(record.get("metrics", {}))
        rows.append(row)
    return rows


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def benchmark_table(rows, metrics=TABLE_METRICS):
    """Markdown table averaging each metric per model. Missing metrics -> '-'."""
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)

    header = (
        "| Model | "
        + " | ".join(f"{m} " + ("↓" if m in ("der", "jer") else "↑") for m in metrics)
        + " |"
    )
    separator = "|---" * (len(metrics) + 1) + "|"
    lines = [header, separator]
    for model in sorted(by_model):
        cells = []
        for metric in metrics:
            values = [r[metric] for r in by_model[model] if isinstance(r.get(metric), (int, float))]
            cells.append(f"{_mean(values):.2f}" if values else "-")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pareto_frontier(rows, quality_key, cost_key, higher_quality_is_better=True):
    """Pareto-optimal rows: no other row is better on both quality and cost."""
    candidates = [
        {k: r[k] for k in ("model", quality_key, cost_key)}
        for r in rows
        if isinstance(r.get(cost_key), (int, float))
        and isinstance(r.get(quality_key), (int, float))
    ]
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            q_better = (
                other[quality_key] > candidate[quality_key]
                if higher_quality_is_better
                else other[quality_key] < candidate[quality_key]
            )
            c_better = other[cost_key] < candidate[cost_key]
            q_equal = other[quality_key] == candidate[quality_key]
            c_equal = other[cost_key] == candidate[cost_key]
            strictly_better_somewhere = q_better or c_better
            not_worse_anywhere = (q_better or q_equal) and (c_better or c_equal)
            if not_worse_anywhere and strictly_better_somewhere:
                dominated = True
                break
            # tie on both: keep only one deterministically
            if q_equal and c_equal and other["model"] < candidate["model"]:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda r: r["model"])


def worst_k(rows, metric, k=5):
    """The k rows with the lowest value of `metric` (failure catalog)."""
    scored = [r for r in rows if isinstance(r.get(metric), (int, float))]
    return sorted(scored, key=lambda r: r[metric])[:k]


RQ_STATUS = {
    "RQ1": "diarization on simultaneous conversations — DER/JER per overlap config",
    "RQ2": "separation vs attribution improvement — hybrid vs diarization-only runs",
    "RQ3": "best pretrained architecture — benchmark table",
    "RQ4": "overlap-aware selective separation trade-off — selective vs full timings",
    "RQ5": "speaker conditioning vs blind attribution — spectral vs embedding runs",
    "RQ6": "degradation with speaker count — speaker-count sweep (pending)",
    "RQ7": "degradation with overlap ratio — overlap sweep (pending)",
    "RQ8": "noise/reverberation robustness — SNR/RIR sweeps (pending)",
    "RQ9": "SI-SDR vs WER correlation — requires ASR evaluation (not implemented)",
}


def render_report(results_dir):
    """Markdown report: benchmark table, Pareto analysis, failures, RQ status."""
    rows = aggregate(results_dir)
    lines = ["# Benchmark Report", ""]

    if not rows:
        lines.append("No results found in this directory.")
        return "\n".join(lines)

    lines += ["## Results", "", benchmark_table(rows), ""]

    pareto_rows = pareto_frontier(
        aggregate_per_model(rows), quality_key="si_sdr", cost_key="runtime"
    )
    lines += ["## Quality / cost Pareto (SI-SDR vs runtime)", ""]
    if pareto_rows:
        lines.append("| Model | SI-SDR | Runtime |")
        lines.append("|---|---|---|")
        for r in pareto_rows:
            lines.append(f"| {r['model']} | {r['si_sdr']:.2f} | {r['runtime']:.3f} |")
    else:
        lines.append("_No runtime data recorded yet._")
    lines.append("")

    lines += ["## Worst cases (SI-SDR)", "", "| Model | Experiment | SI-SDR |", "|---|---|---|"]
    for r in worst_k(rows, "si_sdr", k=5):
        lines.append(f"| {r['model']} | {r.get('experiment', '-')} | {r['si_sdr']:.2f} |")
    lines.append("")

    lines += ["## Research questions", ""]
    for rq, status in RQ_STATUS.items():
        lines.append(f"- **{rq}** — {status}")
    lines.append("")
    lines.append("_Generated from result JSONs; regenerate with `python -m benchmark report`._")
    return "\n".join(lines)


def aggregate_per_model(rows):
    """Average metrics per model for Pareto analysis."""
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    averaged = []
    for model, group in by_model.items():
        entry = {"model": model}
        for key in ("si_sdr", "runtime"):
            values = [r[key] for r in group if isinstance(r.get(key), (int, float))]
            if values:
                entry[key] = _mean(values)
        averaged.append(entry)
    return averaged


def save_report(report, out_path):
    Path(out_path).write_text(report + "\n")
