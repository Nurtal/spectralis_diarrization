"""Extended benchmark matrix: speakers x overlap x SNR on real speech.

Generates datasets under data/benchmarks/matrix_v1/ and evaluates
nmf / sepformer (or sepformer3) / hybrid-spectral on each cell.

Usage:
    uv run python scripts/run_matrix_benchmark.py
"""

import json
import sys
from pathlib import Path

from benchmark.cli import main
from benchmark.datasets import ManifestDataset
from benchmark.generator import generate_dataset
from benchmark.interfaces import Segment
from benchmark.registry import DIARIZERS

BENCH_ROOT = Path("data/benchmarks/matrix_v1")
RESULTS = BENCH_ROOT / "results"
CLIPS = Path("data/clips_testclean")

SPEAKER_COUNTS = (2, 3)
OVERLAPS = (0.0, 0.5)
SNRS = (None, 10)  # None = clean
NUM_MIXTURES = 4
DURATION = 8.0


def cell_name(num_speakers, overlap, snr):
    snr_tag = "clean" if snr is None else f"{snr}db"
    return f"{num_speakers}spk_ovl{int(overlap * 100):02d}_{snr_tag}"


def install_oracle_diarizer(manifest_path):
    dataset = ManifestDataset(manifest_path)
    state = {"call": 0}

    class Oracle:
        def __init__(self, segments):
            self.segments = segments

        def diarize(self, audio, sample_rate):
            return [Segment(s.start, s.end, s.speaker) for s in self.segments]

    def factory(**_):
        mixture = dataset[state["call"] % len(dataset)]
        state["call"] += 1
        return Oracle(mixture.segments)

    DIARIZERS["oracle"] = factory


def neural_model_name(num_speakers):
    return "sepformer" if num_speakers == 2 else "sepformer3"


def main_run():
    RESULTS.mkdir(parents=True, exist_ok=True)

    for num_speakers in SPEAKER_COUNTS:
        for overlap in OVERLAPS:
            for snr in SNRS:
                name = cell_name(num_speakers, overlap, snr)
                manifest = generate_dataset(
                    CLIPS,
                    output_dir=BENCH_ROOT / name,
                    num_mixtures=NUM_MIXTURES,
                    durations=[DURATION],
                    speaker_counts=[num_speakers],
                    overlap_ratios=[overlap],
                    snr_values=[snr],
                    seed=hash(name) % 100000,
                ) if not (BENCH_ROOT / name / "manifest.json").exists() else \
                    BENCH_ROOT / name / "manifest.json"

                runs = [
                    ("nmf", "nmf", {}),
                    ("hybrid", "hybrid", {"diarizer": "oracle", "separator": "nmf",
                                          "attribution": "spectral"}),
                    ("neural", neural_model_name(num_speakers), {}),
                ]
                install_oracle_diarizer(manifest)
                for prefix, model, params in runs:
                    cfg_path = BENCH_ROOT / f"cfg_{prefix}_{name}.yaml"
                    params_lines = "".join(
                        f"      {k}: {json.dumps(v)}\n" for k, v in params.items()
                    )
                    cfg_path.write_text(
                        f"name: {prefix}_{name}\n"
                        "seed: 0\n"
                        f"model:\n  name: {model}\n  params:\n{params_lines}"
                        f"dataset:\n  manifest: {manifest}\n"
                    )
                    rc = main(["evaluate", "--config", str(cfg_path),
                               "--results", str(RESULTS)])
                    print(f"[{name}] {prefix}: rc={rc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main_run())
