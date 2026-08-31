import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from benchmark.audio import save_audio
from benchmark.cli import main


@pytest.fixture
def wav(tmp_path):
    path = tmp_path / "in.wav"
    rng = np.random.default_rng(0)
    save_audio(path, rng.uniform(-0.1, 0.1, 8000).astype(np.float32), 16000)
    return path


class TestHelp:
    def test_module_help_lists_subcommands(self):
        proc = subprocess.run(
            [sys.executable, "-m", "benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0
        for cmd in ("diarize", "separate", "evaluate", "compare", "report", "visualize"):
            assert cmd in proc.stdout


class TestSeparate:
    def test_identity_model_writes_source_files(self, tmp_path, wav):
        out_dir = tmp_path / "out"
        rc = main(
            ["separate", "--model", "identity", "--input", str(wav), "--output", str(out_dir)]
        )
        assert rc == 0
        files = sorted(out_dir.glob("*.wav"))
        assert len(files) == 1
        original, sr = sf.read(wav, dtype="float32")
        saved, sr2 = sf.read(files[0], dtype="float32")
        np.testing.assert_allclose(saved, original)

    def test_unknown_model_fails_with_error(self, tmp_path, wav, capsys):
        rc = main(["separate", "--model", "nope", "--input", str(wav), "--output", str(tmp_path)])
        assert rc != 0
        assert "nope" in capsys.readouterr().err


class TestDiarize:
    def test_noop_diarizer_writes_segments_json(self, tmp_path, wav):
        out = tmp_path / "segments.json"
        rc = main(["diarize", "--input", str(wav), "--output", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert isinstance(data["segments"], list)


class TestEvaluate:
    def test_config_produces_result_json(self, tmp_path):
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            "name: smoke\nseed: 3\nmodel:\n  name: future-model\ndataset:\n  manifest: m.json\n"
        )
        results = tmp_path / "results"

        rc = main(["evaluate", "--config", str(cfg), "--results", str(results)])

        assert rc == 0
        files = list(results.glob("*.json"))
        assert len(files) == 1
        record = json.loads(files[0].read_text())
        assert record["model"] == "future-model"
        assert record["status"] == "not-implemented"


class TestCompare:
    def test_prints_markdown_benchmark_table(self, tmp_path, capsys):
        from benchmark.results import ResultWriter

        w = ResultWriter(tmp_path)
        w.write({"model": "nmf", "metrics": {"si_sdr": 3.1}})
        w.write({"model": "identity", "metrics": {"si_sdr": 0.0}})

        rc = main(["compare", "--results", str(tmp_path)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "| Model |" in out
        assert "nmf" in out and "identity" in out

    def test_compare_empty_results_message(self, tmp_path, capsys):
        rc = main(["compare", "--results", str(tmp_path)])
        assert rc == 0
        assert "no results" in capsys.readouterr().out.lower()


class TestReport:
    def test_writes_markdown_report(self, tmp_path):
        from benchmark.results import ResultWriter

        ResultWriter(tmp_path).write({"model": "nmf", "metrics": {"si_sdr": 3.1}, "runtime": 0.5})
        out = tmp_path / "report.md"

        rc = main(["report", "--results", str(tmp_path), "--out", str(out)])

        assert rc == 0
        content = out.read_text()
        assert "# Benchmark Report" in content
        assert "RQ" in content


class TestVisualize:
    def test_generates_timeline_and_heatmap(self, tmp_path):
        pytest.importorskip("matplotlib")
        # build a tiny manifest with 2 mixtures
        sr = 8000
        rng = np.random.default_rng(1)

        def _wav(name, dur_s=2.0):
            p = tmp_path / name
            save_audio(p, rng.uniform(-0.1, 0.1, int(sr * dur_s)).astype(np.float32), sr)
            return p

        _wav("mix0.wav")
        _wav("srcA0.wav")
        _wav("srcB0.wav")
        _wav("mix1.wav")
        _wav("srcA1.wav")
        _wav("srcB1.wav")

        manifest = {
            "version": "test-v1",
            "sample_rate": sr,
            "mixtures": [
                {
                    "id": "mix_000",
                    "mixture": "mix0.wav",
                    "sources": ["srcA0.wav", "srcB0.wav"],
                    "segments": [
                        {"start": 0.0, "end": 1.0, "speaker": "A"},
                        {"start": 0.8, "end": 2.0, "speaker": "B"},
                    ],
                    "metadata": {"num_speakers": 2},
                },
                {
                    "id": "mix_001",
                    "mixture": "mix1.wav",
                    "sources": ["srcA1.wav", "srcB1.wav"],
                    "segments": [
                        {"start": 0.0, "end": 2.0, "speaker": "A"},
                    ],
                    "metadata": {"num_speakers": 1},
                },
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        out_dir = tmp_path / "viz"
        rc = main(
            [
                "visualize",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--diarizer",
                "vad",
                "--out",
                str(out_dir),
                "--num-samples",
                "2",
                "--seed",
                "0",
            ]
        )
        assert rc == 0
        # 2 timelines + 2 heatmaps + 1 breakdown = 5 pngs
        pngs = list(out_dir.glob("*.png"))
        assert len(pngs) == 5
        assert (out_dir / "timeline_mix_000.png").exists()
        assert (out_dir / "overlap_mix_001.png").exists()
        assert (out_dir / "der_breakdown.png").exists()
        for p in pngs:
            assert p.stat().st_size > 0

    def test_unknown_diarizer_fails(self, tmp_path, capsys):
        # minimal manifest to hit diarizer check first
        sr = 8000
        p = tmp_path / "mix.wav"
        save_audio(p, np.zeros(8000, dtype=np.float32), sr)
        manifest = {
            "version": "test-v1",
            "sample_rate": sr,
            "mixtures": [
                {
                    "id": "mix_000",
                    "mixture": "mix.wav",
                    "sources": ["mix.wav"],
                    "segments": [],
                    "metadata": {},
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        rc = main(
            [
                "visualize",
                "--manifest",
                str(tmp_path / "manifest.json"),
                "--diarizer",
                "nope",
                "--out",
                str(tmp_path / "out"),
            ]
        )
        assert rc != 0
        assert "nope" in capsys.readouterr().err
