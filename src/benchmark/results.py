import json
import time
from pathlib import Path


def _timestamp():
    return time.strftime("%Y%m%dT%H%M%S")


def _slug(text, fallback="model"):
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text)]
    slug = "".join(keep).strip("-")
    return slug or fallback


class ResultWriter:
    """Writes one machine-readable JSON per experiment run (ADR-008)."""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, record):
        if "model" not in record:
            raise ValueError("result record must contain a 'model' field")
        record = dict(record)
        record.setdefault("timestamp", _time_iso())
        base = f"{_slug(record['model'])}_{record['timestamp']}"
        path = self.output_dir / f"{base}.json"
        counter = 1
        while path.exists():
            counter += 1
            path = self.output_dir / f"{base}_{counter}.json"
        path.write_text(json.dumps(record, indent=2) + "\n")
        return path


def read_results(results_dir):
    """Load all result JSONs from a directory, sorted by timestamp."""
    paths = sorted(
        Path(results_dir).glob("*.json"),
        key=lambda p: json.loads(p.read_text()).get("timestamp", ""),
    )
    return [json.loads(p.read_text()) for p in paths]


def _time_iso():
    from datetime import datetime

    return datetime.now().isoformat(timespec="microseconds")
