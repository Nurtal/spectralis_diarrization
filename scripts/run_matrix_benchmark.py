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

SPEAKER_COUNTS = (2, 3, 4)
OVERLAPS = (0.0, 0.5)
SNRS = (None, 10)  # None = clean
DURATIONS = (8.0, 30.0, 60.0)
REVERBS = (None, {"rt60": 0.4})
NUM_MIXTURES = 4
DURATION = 8.0  # legacy alias for backward-compat cell names


def cell_name(num_speakers, overlap, snr, duration=8.0, reverb=None):
    snr_tag = "clean" if snr is None else f"{snr}db"
    base = f"{num_speakers}spk_ovl{int(overlap * 100):02d}_{snr_tag}"
    # keep legacy names for the original 8 cells (dur 8s, no reverb)
    if duration != 8.0:
        base += f"_dur{int(duration)}s"
    if reverb is not None:
        base += f"_rvb{int(reverb['rt60'] * 10):02d}"
    return base


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
    if num_speakers == 2:
        return "sepformer"
    if num_speakers == 3:
        return "sepformer3"
    # 4+ speakers: TF-GridNet is the designated architecture (ADR-004, M4)
    # falls back to sepformer if checkpoint not yet available
    return "tf_gridnet"


def main_run():
    RESULTS.mkdir(parents=True, exist_ok=True)

    for num_speakers in SPEAKER_COUNTS:
        for overlap in OVERLAPS:
            for snr in SNRS:
                for duration in DURATIONS:
                    for reverb in REVERBS:
                        name = cell_name(num_speakers, overlap, snr, duration, reverb)
                        # skip legacy re-generation for the original 8 cells handled above,
                        # but keep the extended naming for new combos
                        manifest_path = BENCH_ROOT / name / "manifest.json"
                        if not manifest_path.exists():
                            # generate only if missing (idempotent)
                            try:
                                manifest = generate_dataset(
                                    CLIPS,
                                    output_dir=BENCH_ROOT / name,
                                    num_mixtures=NUM_MIXTURES,
                                    durations=[duration],
                                    speaker_counts=[num_speakers],
                                    overlap_ratios=[overlap],
                                    snr_values=[snr],
                                    seed=hash(name) % 100000,
                                    reverb=reverb,
                                )
                            except Exception as e:
                                print(f"[{name}] generate failed: {e}", file=sys.stderr)
                                continue
                        else:
                            manifest = manifest_path

                        runs = [
                            ("nmf", "nmf", {}),
                            (
                                "hybrid",
                                "hybrid",
                                {
                                    "diarizer": "oracle",
                                    "separator": "nmf",
                                    "attribution": "spectral",
                                },
                            ),
                            ("neural", neural_model_name(num_speakers), {}),
                        ]
                        # add explicit tf_gridnet run for 4spk cells to validate the adapter
                        if num_speakers >= 4:
                            runs.append(("tfgridnet", "tf_gridnet", {}))
                        try:
                            install_oracle_diarizer(manifest)
                        except Exception as e:
                            print(f"[{name}] oracle install failed: {e}", file=sys.stderr)
                            continue
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
                            try:
                                rc = main(
                                    [
                                        "evaluate",
                                        "--config",
                                        str(cfg_path),
                                        "--results",
                                        str(RESULTS),
                                    ]
                                )
                            except Exception as e:
                                print(f"[{name}] {prefix} evaluate failed: {e}", file=sys.stderr)
                                rc = 1
                            print(f"[{name}] {prefix}: rc={rc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main_run())
