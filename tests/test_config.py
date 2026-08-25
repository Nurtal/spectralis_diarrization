import pytest

from benchmark.config import ExperimentConfig

VALID = """
name: sepformer-smoke
seed: 7
model:
  name: sepformer
  params:
    num_speakers: 2
dataset:
  manifest: data/bench_v1/manifest.json
"""


class TestExperimentConfig:
    def test_loads_valid_yaml(self, tmp_path):
        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text(VALID)

        cfg = ExperimentConfig.load(cfg_file)

        assert cfg.name == "sepformer-smoke"
        assert cfg.seed == 7
        assert cfg.model["name"] == "sepformer"
        assert cfg.model["params"]["num_speakers"] == 2
        assert cfg.dataset["manifest"] == "data/bench_v1/manifest.json"

    def test_seed_defaults_to_zero(self, tmp_path):
        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text("name: x\nmodel:\n  name: nmf\ndataset:\n  manifest: m.json\n")

        cfg = ExperimentConfig.load(cfg_file)

        assert cfg.seed == 0

    def test_missing_required_key_raises_listing_it(self, tmp_path):
        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text("name: x\n")

        with pytest.raises(ValueError, match="model"):
            ExperimentConfig.load(cfg_file)

    def test_unknown_top_level_key_rejected(self, tmp_path):
        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text(
            "name: x\nmodel:\n  name: nmf\ndataset:\n  manifest: m.json\nmodle: 1\n"
        )

        with pytest.raises(ValueError, match="modle"):
            ExperimentConfig.load(cfg_file)

    def test_to_dict_roundtrip(self, tmp_path):
        import yaml

        cfg_file = tmp_path / "exp.yaml"
        cfg_file.write_text(VALID)

        cfg = ExperimentConfig.load(cfg_file)
        reloaded = yaml.safe_load(yaml.safe_dump(cfg.to_dict()))

        assert reloaded["name"] == cfg.name
        assert reloaded["model"] == cfg.model
