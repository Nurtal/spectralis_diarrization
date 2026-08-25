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
        for cmd in ("diarize", "separate", "evaluate", "compare"):
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
    def test_prints_model_names_from_results(self, tmp_path, capsys):
        from benchmark.results import ResultWriter

        w = ResultWriter(tmp_path)
        w.write({"model": "nmf", "si_sdr": 3.1})
        w.write({"model": "identity", "si_sdr": 0.0})

        rc = main(["compare", "--results", str(tmp_path)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "nmf" in out and "identity" in out
