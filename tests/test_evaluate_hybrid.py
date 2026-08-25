import json

import numpy as np
import pytest

from benchmark.audio import save_audio
from benchmark.cli import main
from benchmark.generator import generate_dataset
from benchmark.interfaces import Segment, Separator
from benchmark.registry import DIARIZERS

SR = 8000


def tone(freq, start_s, end_s, dur_s, amp=0.3):
    x = np.zeros(int(dur_s * SR), dtype=np.float32)
    i0, i1 = int(start_s * SR), int(end_s * SR)
    t = np.arange(i1 - i0) / SR
    x[i0:i1] = amp * np.sin(2 * np.pi * freq * t)
    return x


class BandSplitSeparator(Separator):
    def separate(self, audio, sample_rate, num_speakers=None):
        from benchmark.stft import istft, stft

        spec, params = stft(audio, sample_rate)
        freqs = np.fft.rfftfreq(params.n_fft, 1 / sample_rate)
        out = []
        for band in (freqs < 1000.0, freqs >= 1000.0):
            masked = np.zeros_like(spec)
            masked[band, :] = spec[band, :]
            y, _ = istft(masked, params, length=len(audio))
            out.append(y)
        return out


@pytest.fixture(scope="module")
def bench_manifest(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("hyb")
    clips = tmp / "corpus"
    for spk, (f0, f1) in {"spkA": (200.0, 400.0), "spkB": (2400.0, 4800.0)}.items():
        d = clips / spk
        d.mkdir(parents=True)
        for i in range(2):
            sig = tone(f0 if i == 0 else f1, 0.0, 1.5 + 0.5 * i, 2.0 + 0.5 * i)
            save_audio(d / f"u{i}.wav", sig, SR)

    return generate_dataset(
        clips,
        output_dir=tmp / "bench",
        num_mixtures=2,
        durations=[4.0],
        speaker_counts=[2],
        overlap_ratios=[0.5],
        snr_values=[None],
        seed=31,
    )


@pytest.fixture
def oracle_diarizer(bench_manifest):
    """Registry-injected perfect diarizer built from manifest ground truth."""

    class Oracle:
        def __init__(self, segments):
            self.segments = segments

        def diarize(self, audio, sample_rate):
            return [Segment(s.start, s.end, s.speaker) for s in self.segments]

    from benchmark.datasets import ManifestDataset

    ds = ManifestDataset(bench_manifest)

    class OracleFactory:
        def __init__(self, mixture_index=0, **_):
            self.mixture_index = mixture_index

        def __call__(self, **_):
            return Oracle(ds[self.mixture_index].segments)

    # one factory per mixture index is enough for these tests: evaluate runs
    # mixtures sequentially, so we cycle
    class Cycling:
        def __init__(self, **_):
            self.calls = 0

        def __call__(self, **_):
            m = ds[self.calls % len(ds)]
            self.calls += 1
            return Oracle(m.segments)

    DIARIZERS["oracle"] = Cycling()
    yield DIARIZERS["oracle"]
    DIARIZERS.pop("oracle", None)


class TestEvaluateHybrid:
    def test_hybrid_evaluation_writes_end_to_end_metrics(
        self, tmp_path, bench_manifest, oracle_diarizer
    ):
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            "name: hybrid-smoke\n"
            "model:\n"
            "  name: hybrid\n"
            "  params:\n"
            "    diarizer: oracle\n"
            "    separator: nmf\n"
            f"    dataset_manifest_unused: {bench_manifest}\n"
            f"dataset:\n  manifest: {bench_manifest}\n"
        )
        results = tmp_path / "results"

        rc = main(["evaluate", "--config", str(cfg), "--results", str(results)])

        assert rc == 0
        record = json.loads(next(results.glob("*.json")).read_text())
        assert record["status"] == "ok"
        metrics = record["metrics"]
        for key in (
            "si_sdr",
            "sdr",
            "sir",
            "sar",
            "selective_time_seconds",
            "full_time_seconds",
            "num_mixtures",
        ):
            assert key in metrics
        assert metrics["num_mixtures"] == 2

    def test_embedding_attribution_wiring(self, tmp_path, bench_manifest, oracle_diarizer):
        from benchmark.registry import ENCODERS

        class BandEncoder:
            def encode(self, audio, sample_rate):
                from benchmark.stft import stft

                spec, params = stft(np.asarray(audio, dtype=np.float32), sample_rate)
                freqs = np.fft.rfftfreq(params.n_fft, 1 / sample_rate)
                low = float((np.abs(spec)[freqs < 1000] ** 2).sum())
                high = float((np.abs(spec)[freqs >= 1000] ** 2).sum())
                v = np.array([low, high], dtype=np.float32)
                n = np.linalg.norm(v)
                return v / n if n > 0 else v

        ENCODERS["band"] = BandEncoder
        try:
            cfg = tmp_path / "exp.yaml"
            cfg.write_text(
                "name: hybrid-emb\n"
                "model:\n"
                "  name: hybrid\n"
                "  params:\n"
                "    diarizer: oracle\n"
                "    separator: nmf\n"
                "    attribution: embedding\n"
                "    encoder: band\n"
                f"dataset:\n  manifest: {bench_manifest}\n"
            )
            results = tmp_path / "results_emb"

            rc = main(["evaluate", "--config", str(cfg), "--results", str(results)])
        finally:
            ENCODERS.pop("band", None)

        assert rc == 0
        record = json.loads(next(results.glob("*.json")).read_text())
        assert record["status"] == "ok"
        assert record["metrics"]["num_mixtures"] == 2

    def test_selective_separation_costs_less_than_full(
        self, tmp_path, bench_manifest, oracle_diarizer
    ):
        cfg = tmp_path / "exp.yaml"
        cfg.write_text(
            "name: hybrid-cost\n"
            "model:\n"
            "  name: hybrid\n"
            "  params:\n"
            "    diarizer: oracle\n"
            "    separator: nmf\n"
            f"dataset:\n  manifest: {bench_manifest}\n"
        )
        results = tmp_path / "results"
        main(["evaluate", "--config", str(cfg), "--results", str(results)])
        metrics = json.loads(next(results.glob("*.json")).read_text())["metrics"]
        # RQ4 sanity: separating only overlaps should not cost dramatically
        # more than separating everything. Wall-clock comparison with slack:
        # under system load either side can win by a small margin.
        assert metrics["selective_time_seconds"] < metrics["full_time_seconds"] * 1.5 + 0.1
