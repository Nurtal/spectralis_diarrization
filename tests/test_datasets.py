import json

import numpy as np
import pytest
import soundfile as sf

from benchmark.audio import save_audio
from benchmark.datasets import ManifestDataset, Mixture


@pytest.fixture
def dataset_root(tmp_path):
    sr = 8000
    rng = np.random.default_rng(42)

    save_audio(tmp_path / "mix_000.wav", rng.uniform(-0.1, 0.1, 16000).astype(np.float32), sr)
    save_audio(tmp_path / "src_A.wav", rng.uniform(-0.1, 0.1, 16000).astype(np.float32), sr)
    save_audio(tmp_path / "src_B.wav", rng.uniform(-0.1, 0.1, 16000).astype(np.float32), sr)

    manifest = {
        "version": "test-v1",
        "sample_rate": sr,
        "mixtures": [
            {
                "id": "mix_000",
                "mixture": "mix_000.wav",
                "sources": ["src_A.wav", "src_B.wav"],
                "segments": [
                    {"start": 0.0, "end": 1.0, "speaker": "A"},
                    {"start": 0.5, "end": 2.0, "speaker": "B"},
                ],
                "metadata": {"num_speakers": 2, "overlap_ratio": 0.5},
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


class TestMixture:
    def test_loads_mixture_audio(self, dataset_root):
        m = ManifestDataset(dataset_root / "manifest.json")[0]
        audio, sr = m.load_mixture()
        assert sr == 8000
        assert audio.dtype == np.float32
        assert len(audio) == 16000

    def test_loads_ground_truth_sources_in_order(self, dataset_root):
        m = ManifestDataset(dataset_root / "manifest.json")[0]
        sources = [m.load_source(i) for i in range(m.num_sources)]
        assert len(sources) == 2
        direct_a, sr = sf.read(dataset_root / "src_A.wav", dtype="float32")
        np.testing.assert_allclose(sources[0][0], direct_a)
        assert all(s[1] == 8000 for s in sources)


class TestManifestDataset:
    def test_len_and_iteration(self, dataset_root):
        ds = ManifestDataset(dataset_root / "manifest.json")
        assert len(ds) == 1
        items = list(ds)
        assert len(items) == 1
        assert isinstance(items[0], Mixture)

    def test_exposes_metadata_and_segments(self, dataset_root):
        m = ManifestDataset(dataset_root / "manifest.json")[0]
        assert m.mixture_id == "mix_000"
        assert m.metadata["num_speakers"] == 2
        from benchmark.interfaces import Segment

        assert m.segments == (Segment(0.0, 1.0, "A"), Segment(0.5, 2.0, "B"))

    def test_index_out_of_range_raises(self, dataset_root):
        with pytest.raises(IndexError):
            ManifestDataset(dataset_root / "manifest.json")[5]

    def test_missing_file_raises_with_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="manifest"):
            ManifestDataset(tmp_path / "nope.json")
