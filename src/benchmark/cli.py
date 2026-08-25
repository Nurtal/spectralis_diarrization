import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from benchmark.config import ExperimentConfig
from benchmark.registry import DIARIZERS, ENCODERS, SEPARATORS
from benchmark.results import ResultWriter


def build_parser():
    parser = argparse.ArgumentParser(prog="benchmark", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_diar = sub.add_parser("diarize", help="Run a diarizer on an audio file")
    p_diar.add_argument("--input", required=True)
    p_diar.add_argument("--output", required=True)
    p_diar.add_argument("--diarizer", default="noop")

    p_sep = sub.add_parser("separate", help="Separate an audio file into sources")
    p_sep.add_argument("--model", required=True)
    p_sep.add_argument("--input", required=True)
    p_sep.add_argument("--output", required=True)
    p_sep.add_argument("--num-speakers", type=int, default=None)

    p_eval = sub.add_parser("evaluate", help="Run one experiment from a config")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--results", default="results")

    p_cmp = sub.add_parser("compare", help="Summarize result JSONs as a benchmark table")
    p_cmp.add_argument("--results", default="results")

    p_rep = sub.add_parser("report", help="Generate a markdown benchmark report")
    p_rep.add_argument("--results", default="results")
    p_rep.add_argument("--out", default="report.md")

    return parser


def cmd_separate(args):
    from benchmark.audio import load_audio, save_audio

    if args.model not in SEPARATORS:
        available = ", ".join(SEPARATORS)
        print(f"error: unknown model '{args.model}' (available: {available})", file=sys.stderr)
        return 2
    separator = SEPARATORS[args.model]()
    audio, sr = load_audio(args.input)
    sources = separator.separate(audio, sr, num_speakers=args.num_speakers)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, source in enumerate(sources):
        save_audio(out_dir / f"source_{i}.wav", source, sr)
    return 0


def cmd_diarize(args):
    from benchmark.audio import load_audio

    if args.diarizer not in DIARIZERS:
        available = ", ".join(DIARIZERS)
        print(
            f"error: unknown diarizer '{args.diarizer}' (available: {available})",
            file=sys.stderr,
        )
        return 2
    diarizer = DIARIZERS[args.diarizer]()
    audio, sr = load_audio(args.input)
    segments = diarizer.diarize(audio, sr)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diarizer": args.diarizer,
        "segments": [asdict(s) for s in segments],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return 0


def cmd_evaluate(args):
    cfg = ExperimentConfig.load(args.config)
    model_name = cfg.model["name"]

    if model_name == "hybrid" and "manifest" in cfg.dataset:
        record = evaluate_hybrid(cfg)
    elif model_name in DIARIZERS and "manifest" in cfg.dataset:
        record = evaluate_diarization(cfg)
    elif model_name in SEPARATORS and "manifest" in cfg.dataset:
        record = evaluate_separation(cfg)
    else:
        record = {
            "experiment": cfg.name,
            "seed": cfg.seed,
            "model": model_name,
            "model_config": cfg.model,
            "dataset": cfg.dataset,
            "metrics": {},
            "status": "not-implemented",
        }
    ResultWriter(args.results).write(record)
    print(f"experiment '{cfg.name}' recorded to {args.results}/")
    return 0


def evaluate_hybrid(cfg):
    """End-to-end hybrid pipeline evaluation: selective separation, attribution,
    reassembly, scored per speaker against ground truth (ADR-005, RQ4)."""
    from benchmark.datasets import ManifestDataset
    from benchmark.hybrid import HybridPipeline
    from benchmark.separation_metrics import bss_metrics, si_sdr

    dataset = ManifestDataset(cfg.dataset["manifest"])
    params = cfg.model.get("params") or {}
    diarizer = DIARIZERS[params.get("diarizer", "noop")](**params.get("diarizer_params", {}))
    separator = SEPARATORS[params.get("separator", "identity")](
        **params.get("separator_params", {})
    )
    attribution = params.get("attribution", "spectral")
    encoder = None
    if attribution == "embedding":
        encoder_name = params.get("encoder")
        if encoder_name is None:
            print("error: hybrid embedding attribution requires params.encoder", file=sys.stderr)
            return 2
        encoder = ENCODERS[encoder_name](**params.get("encoder_params", {}))
    pipeline = HybridPipeline(
        diarizer=diarizer, separator=separator, attribution=attribution, encoder=encoder
    )

    totals = {"si_sdr": 0.0, "sdr": 0.0, "sir": 0.0, "sar": 0.0}
    selective_total = full_total = 0.0
    n = 0
    for mixture in dataset:
        audio, sr = mixture.load_mixture()
        result = pipeline.process(
            audio,
            sr,
            num_speakers=mixture.metadata.get("num_speakers"),
            compare_full=True,
        )

        for src_idx, spk in enumerate(mixture.source_speakers):
            if spk not in result.tracks:
                continue
            reference, _ = mixture.load_source(src_idx)
            totals["si_sdr"] += si_sdr(reference, result.tracks[spk])
            bss = bss_metrics([reference], [result.tracks[spk]])
            for key in ("sdr", "sir", "sar"):
                totals[key] += bss[key]

        selective_total += result.selective_time
        full_total += result.full_time if result.full_time is not None else 0.0
        n += 1

    metrics = {k: v / n for k, v in totals.items()} if n else dict(totals)
    metrics.update(
        selective_time_seconds=selective_total,
        full_time_seconds=full_total,
        num_mixtures=n,
    )
    return {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "model": cfg.model["name"],
        "model_config": cfg.model,
        "dataset": cfg.dataset,
        "metrics": metrics,
        "status": "ok",
    }


def evaluate_diarization(cfg):
    """Run a registered diarizer over a manifest dataset and score DER/JER."""
    from benchmark.datasets import ManifestDataset
    from benchmark.metrics import der, jer, overlap_detection_scores

    dataset = ManifestDataset(cfg.dataset["manifest"])
    diarizer = DIARIZERS[cfg.model["name"]](**(cfg.model.get("params") or {}))

    totals = {"der": 0.0, "jer": 0.0, "overlap_precision": 0.0, "overlap_recall": 0.0}
    n = 0
    for mixture in dataset:
        audio, sr = mixture.load_mixture()
        hypothesis = diarizer.diarize(audio, sr)
        scores = der(list(mixture.segments), hypothesis)
        totals["der"] += scores["der"]
        totals["jer"] += jer(list(mixture.segments), hypothesis)
        ov = overlap_detection_scores(list(mixture.segments), hypothesis)
        totals["overlap_precision"] += ov["precision"]
        totals["overlap_recall"] += ov["recall"]
        n += 1

    metrics = {k: v / n for k, v in totals.items()} if n else dict(totals)
    metrics["num_mixtures"] = n
    return {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "model": cfg.model["name"],
        "model_config": cfg.model,
        "dataset": cfg.dataset,
        "metrics": metrics,
        "status": "ok",
    }


def cmd_compare(args):
    from benchmark.analysis import aggregate, benchmark_table

    rows = aggregate(args.results)
    if not rows:
        print(f"no results found in {args.results}")
        return 0
    print(benchmark_table(rows))
    return 0


def cmd_report(args):
    from benchmark.analysis import render_report, save_report

    report = render_report(args.results)
    out = Path(args.out)
    if str(out.parent) != "":
        out.parent.mkdir(parents=True, exist_ok=True)
    save_report(report, out)
    print(f"report written to {out}")
    return 0


def evaluate_separation(cfg):
    """Run a registered separator over a manifest dataset and score SI-SDR/BSS."""
    import time

    from benchmark.datasets import ManifestDataset
    from benchmark.separation_metrics import best_pairing_si_sdr, bss_metrics

    dataset = ManifestDataset(cfg.dataset["manifest"])
    separator = SEPARATORS[cfg.model["name"]](**(cfg.model.get("params") or {}))

    totals = {"si_sdr": 0.0, "sdr": 0.0, "sir": 0.0, "sar": 0.0}
    inference_time = 0.0
    n = 0
    for mixture in dataset:
        audio, sr = mixture.load_mixture()
        references = [mixture.load_source(i)[0] for i in range(mixture.num_sources)]

        start = time.perf_counter()
        estimates = separator.separate(
            audio,
            sr,
            num_speakers=mixture.metadata.get("num_speakers"),
        )
        inference_time += time.perf_counter() - start

        totals["si_sdr"] += best_pairing_si_sdr(references, estimates)
        bss = bss_metrics(references, estimates)
        for key in ("sdr", "sir", "sar"):
            totals[key] += bss[key]
        n += 1

    metrics = {k: v / n for k, v in totals.items()} if n else dict(totals)
    metrics.update(inference_time_seconds=inference_time, num_mixtures=n)
    return {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "model": cfg.model["name"],
        "model_config": cfg.model,
        "dataset": cfg.dataset,
        "metrics": metrics,
        "status": "ok",
    }


COMMANDS = {
    "separate": cmd_separate,
    "diarize": cmd_diarize,
    "evaluate": cmd_evaluate,
    "compare": cmd_compare,
    "report": cmd_report,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
