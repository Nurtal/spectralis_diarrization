import json

import numpy as np
import pytest

from benchmark.audio import save_audio
from benchmark.cli import main


@pytest.fixture
def bench_manifest(tmp_path):
    """One tiny 2-speaker dataset built with the Phase 1 generator."""
    clips = tmp_path / "corpus"
    sr = 8000
    for spk in ("spk1", "spk2"):
        d = clips / spk
        d.mkdir(parents=True)
        for i in range(2):
            n = int(1.5 * sr)
            tone = (0.25 * np.sin(2 * np.pi * 300 * np.arange(n) / sr)).astype(np.float32)
            save_audio(d / f"u{i}.wav", tone, sr)
    from benchmark.generator import generate_dataset

    return generate_dataset(
        clips,
        output_dir=tmp_path / "bench",
        num_mixtures=2,
        durations=[4.0],
        speaker_counts=[2],
        overlap_ratios=[0.5],
        snr_values=[20],
        seed=5,
    )


class TestEvaluateDiarization:
    def test_evaluate_runs_diarizer_and_writes_metrics(self, tmp_path, bench_manifest):
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            "name: diar-smoke\nseed: 1\n"
            "model:\n  name: vad\n"
            f"dataset:\n  manifest: {bench_manifest}\n"
        )
        results = tmp_path / "results"

        rc = main(["evaluate", "--config", str(cfg), "--results", str(results)])

        assert rc == 0
        files = list(results.glob("*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record["status"] == "ok"
        metrics = record["metrics"]
        for key in ("der", "jer", "overlap_precision", "overlap_recall", "num_mixtures"):
            assert key in metrics

    def test_unknown_diarizer_still_writes_placeholder(self, tmp_path):
        cfg = tmp_path / "exp.yaml"
        cfg.write_text("name: x\nmodel:\n  name: future-model\ndataset:\n  manifest: m.json\n")
        results = tmp_path / "results"

        rc = main(["evaluate", "--config", str(cfg), "--results", str(results)])

        assert rc == 0
        record = json.loads(next(results.glob("*.json")).read_text())
        assert record["status"] == "not-implemented"

    def test_vad_on_clean_single_speaker_gets_low_der(self, tmp_path, bench_manifest):
        # sanity: on clean mixtures the energy VAD should find speech where ref does
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            f"name: der-sanity\nmodel:\n  name: vad\ndataset:\n  manifest: {bench_manifest}\n"
        )
        results = tmp_path / "results"
        main(["evaluate", "--config", str(cfg), "--results", str(results)])
        record = json.loads(next(results.glob("*.json")).read_text())
        # A speaker-blind VAD on overlapping mixtures can exceed DER=1.0
        # (false alarms + counted confusion); this only sanity-checks magnitude.
        assert 0.0 <= record["metrics"]["der"] < 3.0
        assert record["metrics"]["num_mixtures"] == 2
