import matplotlib

matplotlib.use("Agg")

import pytest

from benchmark.analysis import aggregate
from benchmark.plots import plot_metric_by_model, plot_quality_vs_cost, plot_sweep
from benchmark.results import ResultWriter

pytest.importorskip("matplotlib")


@pytest.fixture
def results_dir(tmp_path):
    w = ResultWriter(tmp_path)
    w.write({"model": "nmf", "metrics": {"si_sdr": 7.9}, "metrics_runtime": None, "runtime": 0.5})
    w.write({"model": "sepformer", "metrics": {"si_sdr": 12.4}, "runtime": 4.2})
    return tmp_path


class TestPlots:
    def test_quality_vs_cost_saves_file(self, results_dir, tmp_path):
        out = tmp_path / "pareto.png"
        plot_quality_vs_cost(aggregate(results_dir), quality="si_sdr", cost="runtime", out=out)
        assert out.exists() and out.stat().st_size > 0

    def test_metric_by_model_bar(self, results_dir, tmp_path):
        out = tmp_path / "bars.png"
        plot_metric_by_model(aggregate(results_dir), metric="si_sdr", out=out)
        assert out.exists()

    def test_sweep_curve(self, tmp_path):
        out = tmp_path / "sweep.png"
        rows = [{"model": "nmf", "overlap": o, "si_sdr": 8 - o * 4} for o in (0.0, 0.25, 0.5)] + [
            {"model": "sepformer", "overlap": o, "si_sdr": 12 - o * 3} for o in (0.0, 0.25, 0.5)
        ]
        plot_sweep(rows, x_key="overlap", y_key="si_sdr", out=out)
        assert out.exists()
