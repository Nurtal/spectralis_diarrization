import matplotlib

matplotlib.use("Agg")

import pytest

from benchmark.analysis import aggregate
from benchmark.interfaces import Segment
from benchmark.plots import (
    plot_der_breakdown,
    plot_diarization_timeline,
    plot_metric_by_model,
    plot_overlap_heatmap,
    plot_quality_vs_cost,
    plot_sweep,
)
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


class TestDiarizationViz:
    def test_timeline_saves_file(self, tmp_path):
        ref = [
            Segment(0.0, 2.0, "A"),
            Segment(1.0, 3.0, "B"),
        ]
        hyp = [
            Segment(0.0, 2.2, "A"),
            Segment(1.5, 3.0, "B"),
        ]
        out = tmp_path / "timeline.png"
        plot_diarization_timeline(ref, hyp, out)
        assert out.exists() and out.stat().st_size > 0

    def test_timeline_empty_segments(self, tmp_path):
        out = tmp_path / "timeline_empty.png"
        plot_diarization_timeline([], [], out)
        assert out.exists()

    def test_timeline_handles_4_speakers(self, tmp_path):
        ref = [Segment(i * 0.5, i * 0.5 + 1.0, f"spk{i}") for i in range(4)]
        hyp = [Segment(i * 0.5 + 0.1, i * 0.5 + 1.1, f"spk{i}") for i in range(4)]
        out = tmp_path / "timeline_4spk.png"
        plot_diarization_timeline(ref, hyp, out, title="4spk test")
        assert out.exists()

    def test_overlap_heatmap_saves_file(self, tmp_path):
        ref = [
            Segment(0.0, 3.0, "A"),
            Segment(1.0, 2.0, "B"),
        ]
        hyp = [
            Segment(0.0, 3.0, "A"),
            Segment(1.2, 2.2, "B"),
        ]
        out = tmp_path / "heatmap.png"
        plot_overlap_heatmap(ref, hyp, out)
        assert out.exists() and out.stat().st_size > 0

    def test_overlap_heatmap_empty(self, tmp_path):
        out = tmp_path / "heatmap_empty.png"
        plot_overlap_heatmap([], [], out)
        assert out.exists()

    def test_der_breakdown_saves_file(self, tmp_path):
        metrics = [
            {"der": 0.1, "missed": 0.2, "false_alarm": 0.1, "confusion": 0.05},
            {"der": 0.3, "missed": 0.5, "false_alarm": 0.2, "confusion": 0.1},
            {"der": 0.05, "missed": 0.1, "false_alarm": 0.05, "confusion": 0.02},
        ]
        out = tmp_path / "der_breakdown.png"
        plot_der_breakdown(metrics, out)
        assert out.exists() and out.stat().st_size > 0

    def test_der_breakdown_empty(self, tmp_path):
        out = tmp_path / "der_empty.png"
        plot_der_breakdown([], out)
        assert out.exists()
