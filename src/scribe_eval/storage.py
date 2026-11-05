"""Run directory layout and durable records."""

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from scribnic import load_artifact

from .types import Prediction


def new_run_directory(root: Path, name: str) -> Path:
    """Choose a unique run directory without creating it."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return root / f"{stamp}-{name}-{uuid4().hex[:6]}"


def sha256_text(text: str) -> str:
    """Hash text for comparing configurations."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, data: object) -> None:
    """Replace a JSON file atomically."""
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n"
    _write_atomic(path, text)


def read_json(path: Path) -> dict:
    """Read a JSON object."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, text: str) -> None:
    """Write through a temporary file, so readers never see a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(text)
    Path(stream.name).replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    """Append one row and flush it to disk, so a crash loses at most that row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Replace a JSONL file atomically."""
    lines = [json.dumps(row, ensure_ascii=False, default=str) for row in rows]
    _write_atomic(path, "".join(line + "\n" for line in lines))


def read_jsonl(path: Path) -> list[dict]:
    """Read rows, ignoring a last line cut off by a crash.

    Raises:
        ValueError: If a line other than the last one is invalid.
    """
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines, start=1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number == len(lines):
                break
            raise ValueError(f"Corrupt line {number} in {path}.") from error
    return rows


@dataclass(frozen=True)
class RunPaths:
    """Where each record of a run lives."""

    root: Path

    @property
    def experiment(self) -> Path:
        """Exact copy of the experiment file."""
        return self.root / "experiment.toml"

    @property
    def manifest(self) -> Path:
        """Frozen IDs, versions, and effective workflow."""
        return self.root / "manifest.json"

    @property
    def status(self) -> Path:
        """Progress summary; the JSONL files are the source of truth."""
        return self.root / "status.json"

    @property
    def startup(self) -> Path:
        """Model load times, kept apart from inference."""
        return self.root / "startup.jsonl"

    @property
    def results(self) -> Path:
        """Aggregated metrics."""
        return self.root / "results.json"

    def predictions(self, dataset: str) -> Path:
        """One row per workflow execution."""
        return self.root / "datasets" / dataset / "predictions.jsonl"

    def scores(self, dataset: str) -> Path:
        """One row per metric and prediction."""
        return self.root / "datasets" / dataset / "scores.jsonl"

    def next_attempt(self, dataset: str, sample_id: str) -> Path:
        """Pick an unused directory, because Scribnic requires an empty one."""
        samples = self.root / "datasets" / dataset / "samples" / sample_id
        attempt = 1
        while (samples / f"attempt-{attempt}").exists():
            attempt += 1
        return samples / f"attempt-{attempt}"


def prediction_row(prediction: Prediction) -> dict:
    """Serialize a prediction; artifacts stay in Scribnic's run directory."""
    return {
        "dataset": prediction.dataset,
        "sample_id": prediction.sample_id,
        "variant": prediction.variant,
        "repeat": prediction.repeat,
        "input_kind": prediction.input_kind,
        "status": prediction.status,
        "error": prediction.error,
        "artifacts": sorted(prediction.artifacts),
        "steps": list(prediction.steps),
        "resources": prediction.resources,
        "elapsed_seconds": prediction.elapsed_seconds,
        "run_directory": prediction.run_directory,
    }


def load_prediction(root: Path, row: dict) -> Prediction:
    """Rebuild a prediction, reading its artifacts from Scribnic's run directory."""
    directory = root / row["run_directory"]
    artifacts = {
        kind: load_artifact(directory / f"{kind}.json") for kind in row["artifacts"]
    }
    return Prediction(
        dataset=row["dataset"],
        sample_id=row["sample_id"],
        variant=row["variant"],
        repeat=row["repeat"],
        input_kind=row["input_kind"],
        status=row["status"],
        error=row["error"],
        artifacts=artifacts,
        steps=tuple(row["steps"]),
        resources=row["resources"],
        elapsed_seconds=row["elapsed_seconds"],
        run_directory=row["run_directory"],
    )


def latest_rows(rows: Iterable[dict]) -> list[dict]:
    """Keep the last row of each execution; a retry supersedes an earlier failure."""
    latest = {}
    for row in rows:
        latest[(row["variant"], row["sample_id"], row["repeat"])] = row
    return list(latest.values())
