import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from benchmark.config import ExperimentConfig
from benchmark.registry import DIARIZERS, SEPARATORS
from benchmark.results import ResultWriter, read_results


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

    p_eval = sub.add_parser("evaluate", help="Run one experiment from a config")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--results", default="results")

    p_cmp = sub.add_parser("compare", help="Summarize result JSONs in a directory")
    p_cmp.add_argument("--results", default="results")

    return parser


def cmd_separate(args):
    from benchmark.audio import load_audio, save_audio

    if args.model not in SEPARATORS:
        available = ", ".join(SEPARATORS)
        print(f"error: unknown model '{args.model}' (available: {available})", file=sys.stderr)
        return 2
    separator = SEPARATORS[args.model]()
    audio, sr = load_audio(args.input)
    sources = separator.separate(audio, sr)
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
    record = {
        "experiment": cfg.name,
        "seed": cfg.seed,
        "model": cfg.model["name"],
        "model_config": cfg.model,
        "dataset": cfg.dataset,
        "metrics": {},
        "status": "not-implemented",
    }
    ResultWriter(args.results).write(record)
    print(f"experiment '{cfg.name}' recorded to {args.results}/")
    return 0


def cmd_compare(args):
    records = read_results(args.results)
    if not records:
        print(f"no results found in {args.results}")
        return 0
    models = [r.get("model", "?") for r in records]
    si_sdr = [r.get("si_sdr") for r in records]
    width = max(len(m) for m in models + ["model"])
    print(f"{'model'.ljust(width)}  si_sdr")
    for model, value in zip(models, si_sdr):
        print(f"{model.ljust(width)}  {'-' if value is None else format(value, '.2f')}")
    return 0


COMMANDS = {
    "separate": cmd_separate,
    "diarize": cmd_diarize,
    "evaluate": cmd_evaluate,
    "compare": cmd_compare,
}


def main(argv=None):
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
