import json
from dataclasses import dataclass
from pathlib import Path

from benchmark.audio import load_audio
from benchmark.interfaces import Segment


@dataclass(frozen=True)
class Mixture:
    """A single benchmark mixture with lazy access to audio and ground truth."""

    mixture_id: str
    mixture_path: Path
    source_paths: tuple
    segments: tuple
    metadata: dict
    sample_rate: int

    @property
    def num_sources(self):
        return len(self.source_paths)

    def load_mixture(self):
        return load_audio(self.mixture_path)

    def load_source(self, index):
        if not 0 <= index < self.num_sources:
            raise IndexError(f"source index {index} out of range for {self.num_sources} sources")
        return load_audio(self.source_paths[index])


class ManifestDataset:
    """Dataset backed by a manifest.json produced by the Phase 1 generator."""

    def __init__(self, manifest_path):
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.root = manifest_path.parent
        raw = json.loads(manifest_path.read_text())
        self.version = raw.get("version", "unversioned")
        self.sample_rate = raw["sample_rate"]
        self._mixtures = [
            Mixture(
                mixture_id=entry["id"],
                mixture_path=self.root / entry["mixture"],
                source_paths=tuple(self.root / s for s in entry["sources"]),
                segments=tuple(
                    Segment(start=s["start"], end=s["end"], speaker=s["speaker"])
                    for s in entry.get("segments", [])
                ),
                metadata=dict(entry.get("metadata", {})),
                sample_rate=raw["sample_rate"],
            )
            for entry in raw["mixtures"]
        ]

    def __len__(self):
        return len(self._mixtures)

    def __getitem__(self, index):
        return self._mixtures[index]

    def __iter__(self):
        return iter(self._mixtures)
