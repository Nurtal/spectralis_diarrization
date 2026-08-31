"""Run real-speech benchmark with pyannote diarization + sepformer separation."""

import json
import sys
from pathlib import Path

from benchmark.cli import main
from benchmark.datasets import ManifestDataset
from benchmark.interfaces import Segment
from benchmark.registry import DIARIZERS

BENCH_ROOT = Path("data/benchmarks/real_speech_v1")
RESULTS = BENCH_ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

OVERLAPS = (0, 25, 50, 75)


def install_pyannote_diarizer(manifest_path):
    """Real pyannote diarizer (not oracle)."""
    from benchmark.registry import DIARIZERS as REG

    REG["pyannote_real"] = lambda **kw: DIARIZERS["pyannote"](**kw)


def install_oracle_diarizer(manifest_path):
    """Oracle diarizer cycling through manifest ground truth (upper bound)."""
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


def main_run():
    for overlap in OVERLAPS:
        manifest = BENCH_ROOT / f"ovl{overlap:02d}" / "manifest.json"
        install_pyannote_diarizer(manifest)
        install_oracle_diarizer(manifest)

        runs = [
            # Diarization only
            ("diarization_pyannote", "pyannote_real", {}),
            # Separation only
            ("sepformer", "sepformer", {}),
            # Hybrid: oracle diarization + sepformer separation (best case)
            (
                "hybrid_oracle_sepformer_spectral",
                "hybrid",
                {"diarizer": "oracle", "separator": "sepformer", "attribution": "spectral"},
            ),
            (
                "hybrid_oracle_sepformer_embedding",
                "hybrid",
                {
                    "diarizer": "oracle",
                    "separator": "sepformer",
                    "attribution": "embedding",
                    "encoder": "ecapa",
                },
            ),
            # Hybrid: pyannote diarization + sepformer separation (realistic)
            (
                "hybrid_pyannote_sepformer_spectral",
                "hybrid",
                {"diarizer": "pyannote_real", "separator": "sepformer", "attribution": "spectral"},
            ),
            (
                "hybrid_pyannote_sepformer_embedding",
                "hybrid",
                {
                    "diarizer": "pyannote_real",
                    "separator": "sepformer",
                    "attribution": "embedding",
                    "encoder": "ecapa",
                },
            ),
        ]

        for prefix, model, params in runs:
            cfg_path = BENCH_ROOT / f"cfg_{prefix}_ovl{overlap:02d}.yaml"
            params_lines = "".join(f"      {k}: {json.dumps(v)}\n" for k, v in params.items())
            cfg_path.write_text(
                f"name: {prefix}_ovl{overlap:02d}\n"
                "seed: 0\n"
                f"model:\n  name: {model}\n  params:\n{params_lines}"
                f"dataset:\n  manifest: {manifest}\n"
            )
            rc = main(["evaluate", "--config", str(cfg_path), "--results", str(RESULTS)])
            print(f"[ovl{overlap:02d}] {prefix}: rc={rc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main_run())
