from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one benchmark experiment run."""

    name: str
    model: dict
    dataset: dict
    seed: int = 0
    extra: dict = field(default_factory=dict)

    REQUIRED_KEYS = ("name", "model", "dataset")

    @classmethod
    def load(cls, path):
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"config must be a YAML mapping: {path}")

        missing = [k for k in cls.REQUIRED_KEYS if k not in raw]
        if missing:
            raise ValueError(f"config is missing required key(s): {', '.join(missing)}")

        known = set(cls.REQUIRED_KEYS) | {"seed"}
        unknown = [k for k in raw if k not in known]
        if unknown:
            raise ValueError(f"unknown config key(s): {', '.join(unknown)}")

        return cls(
            name=raw["name"],
            model=dict(raw["model"]),
            dataset=dict(raw["dataset"]),
            seed=int(raw.get("seed", 0)),
        )

    def to_dict(self):
        return {
            "name": self.name,
            "seed": self.seed,
            "model": self.model,
            "dataset": self.dataset,
        }
