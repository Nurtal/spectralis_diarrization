import json

import pytest

from benchmark.results import ResultWriter, read_results


class TestResultWriter:
    def test_write_creates_json_preserving_record(self, tmp_path):
        record = {"model": "nmf", "si_sdr": 3.2}

        path = ResultWriter(tmp_path).write(record)

        loaded = json.loads(path.read_text())
        assert loaded["model"] == "nmf"
        assert loaded["si_sdr"] == 3.2

    def test_rejects_record_without_model(self, tmp_path):
        with pytest.raises(ValueError, match="model"):
            ResultWriter(tmp_path).write({"si_sdr": 1.0})

    def test_stamps_timestamp_automatically(self, tmp_path):
        path = ResultWriter(tmp_path).write({"model": "nmf"})

        loaded = json.loads(path.read_text())
        assert "timestamp" in loaded

    def test_filenames_do_not_collide(self, tmp_path):
        writer = ResultWriter(tmp_path)
        p1 = writer.write({"model": "nmf"})
        p2 = writer.write({"model": "nmf"})

        assert p1 != p2
        assert p2.exists()


class TestReadResults:
    def test_reads_all_json_files_sorted_by_time(self, tmp_path):
        w = ResultWriter(tmp_path)
        w.write({"model": "a"})
        w.write({"model": "b"})

        records = read_results(tmp_path)
        assert [r["model"] for r in records] == ["a", "b"]

    def test_ignores_non_json_files(self, tmp_path):
        (tmp_path / "notes.txt").write_text("hello")
        ResultWriter(tmp_path).write({"model": "a"})

        assert len(read_results(tmp_path)) == 1
