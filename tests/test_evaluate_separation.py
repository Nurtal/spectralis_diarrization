import json

import numpy as np
import pytest

from benchmark.audio import save_audio
from benchmark.cli import main
from benchmark.generator import generate_dataset


@pytest.fixture(scope="module")
def bench_manifest(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sep")
    clips = tmp / "corpus"
    sr = 8000
    for spk_i, freq in enumerate((250.0, 2000.0)):
        d = clips / f"spk{spk_i}"
        d.mkdir(parents=True)
        for i in range(2):
            n = int(1.0 * sr)
            t = np.arange(n) / sr
            tone = (0.25 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
            save_audio(d / f"u{i}.wav", tone, sr)

    return generate_dataset(
        clips,
        output_dir=tmp / "bench",
        num_mixtures=1,
        durations=[3.0],
        speaker_counts=[2],
        overlap_ratios=[0.5],
        snr_values=[20],
        seed=9,
    )


class TestEvaluateSeparation:
    def test_nmf_evaluation_writes_si_sdr_metrics(self, tmp_path, bench_manifest):
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            "name: nmf-smoke\nseed: 0\n"
            "model:\n  name: nmf\n  params:\n    seed: 0\n"
            f"dataset:\n  manifest: {bench_manifest}\n"
        )
        results = tmp_path / "results"

        rc = main(["evaluate", "--config", str(cfg), "--results", str(results)])

        assert rc == 0
        record = json.loads(next(results.glob("*.json")).read_text())
        assert record["status"] == "ok"
        for key in ("si_sdr", "sdr", "sir", "sar", "inference_time_seconds", "num_mixtures"):
            assert key in record["metrics"]

    def test_separate_cli_runs_nmf(self, tmp_path):
        sr = 8000
        wav = tmp_path / "mix.wav"
        n = int(1.5 * sr)
        t = np.arange(n) / sr
        audio = 0.25 * np.sin(2 * np.pi * 300.0 * t) + 0.25 * np.sin(2 * np.pi * 2400.0 * t)
        save_audio(wav, np.clip(audio, -1, 1).astype(np.float32), sr)
        out_dir = tmp_path / "out"

        rc = main(
            [
                "separate",
                "--model",
                "nmf",
                "--num-speakers",
                "2",
                "--input",
                str(wav),
                "--output",
                str(out_dir),
            ]
        )

        assert rc == 0
        files = sorted(out_dir.glob("source_*.wav"))
        assert len(files) == 2
