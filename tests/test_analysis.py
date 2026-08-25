import pytest

from benchmark.analysis import (
    aggregate,
    benchmark_table,
    pareto_frontier,
    render_report,
    worst_k,
)
from benchmark.results import ResultWriter


@pytest.fixture
def results_dir(tmp_path):
    w = ResultWriter(tmp_path)
    w.write({"model": "nmf", "metrics": {"si_sdr": 7.9, "sir": 13.9}, "seed": 0})
    w.write({"model": "sepformer", "metrics": {"si_sdr": 12.4, "sir": 17.2}, "seed": 1})
    w.write({"model": "sepformer", "metrics": {"si_sdr": 10.6, "sir": 16.0}, "seed": 2})
    return tmp_path


class TestAggregate:
    def test_flattens_metrics_into_rows(self, results_dir):
        rows = aggregate(results_dir)
        assert len(rows) == 3
        row = next(r for r in rows if r["model"] == "nmf")
        assert row["si_sdr"] == 7.9
        assert row["sir"] == 13.9
        assert row["seed"] == 0


class TestBenchmarkTable:
    def test_markdown_table_averaged_per_model(self, results_dir):
        table = benchmark_table(aggregate(results_dir))
        assert "| model" in table.lower()
        assert "si_sdr" in table
        # sepformer appears once with averaged si_sdr (12.4+10.6)/2 = 11.5
        sep_row = [line for line in table.splitlines() if "sepformer" in line][0]
        assert "11.50" in sep_row

    def test_missing_metric_rendered_as_dash(self, tmp_path):
        w = ResultWriter(tmp_path)
        w.write({"model": "vad", "metrics": {"der": 1.9}})
        table = benchmark_table(aggregate(tmp_path))
        vad_row = [line for line in table.splitlines() if "vad" in line][0]
        assert "-" in vad_row


class TestParetoFrontier:
    def test_dominated_models_excluded(self):
        rows = [
            {"model": "slow-strong", "si_sdr": 12.0, "runtime": 100},
            {"model": "fast-weak", "si_sdr": 5.0, "runtime": 1},
            {"model": "dominated", "si_sdr": 4.0, "runtime": 50},
            {"model": "balanced", "si_sdr": 8.0, "runtime": 2},
        ]
        frontier = pareto_frontier(rows, quality_key="si_sdr", cost_key="runtime")
        names = {r["model"] for r in frontier}
        assert "dominated" not in names
        assert {"slow-strong", "fast-weak", "balanced"} <= names

    def test_rows_without_cost_are_skipped(self):
        rows = [{"model": "a", "si_sdr": 1.0}, {"model": "b", "si_sdr": 2.0, "runtime": 3}]
        frontier = pareto_frontier(rows, quality_key="si_sdr", cost_key="runtime")
        assert [r["model"] for r in frontier] == ["b"]

    def test_equal_quality_lower_cost_wins(self):
        rows = [
            {"model": "a", "q": 5.0, "c": 10},
            {"model": "b", "q": 5.0, "c": 5},
        ]
        frontier = pareto_frontier(rows, quality_key="q", cost_key="c")
        assert [r["model"] for r in frontier] == ["b"]


class TestWorstK:
    def test_returns_k_lowest_scores(self, results_dir):
        worst = worst_k(aggregate(results_dir), metric="si_sdr", k=2)
        models = [r["model"] for r in worst]
        assert models[0] == "nmf"  # 7.9 lowest
        assert len(worst) == 2

    def test_k_larger_than_rows_returns_all(self, results_dir):
        assert len(worst_k(aggregate(results_dir), metric="si_sdr", k=99)) == 3


class TestRenderReport:
    def test_report_contains_tables_and_rq_section(self, results_dir):
        report = render_report(results_dir)
        assert "# Benchmark Report" in report
        assert "| model" in report.lower()
        assert "RQ" in report
        assert "Pareto" in report or "pareto" in report

    def test_report_from_empty_dir_mentions_no_results(self, tmp_path):
        report = render_report(tmp_path)
        assert "no results" in report.lower()
