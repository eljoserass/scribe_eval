"""Load experiment definitions from TOML."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .artifacts import ArtifactKind, artifact_kind

_EXPERIMENT_KEYS = {"name", "pipeline", "datasets"}
_DATASET_KEYS = {
    "name",
    "split",
    "input",
    "metrics",
    "limit",
    "fraction",
    "ids",
    "seed",
}


@dataclass(frozen=True)
class DatasetEntry:
    """One dataset in an experiment: what enters the workflow and what is measured."""

    name: str
    split: str
    input: ArtifactKind
    metrics: tuple[str, ...]
    limit: int | None = None
    fraction: float | None = None
    ids: tuple[str, ...] | None = None
    seed: int = 42


@dataclass(frozen=True)
class ExperimentConfig:
    """One pipeline evaluated on one or more datasets."""

    name: str
    pipeline_type: str
    pipeline_settings: dict[str, object]
    datasets: tuple[DatasetEntry, ...]


def _check_keys(data: dict, allowed: set[str], label: str) -> None:
    """Reject unknown keys, so a typo cannot silently change an experiment."""
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown {label} keys: {', '.join(unknown)}")


def _dataset_entry(data: dict) -> DatasetEntry:
    """Build and check one `[[datasets]]` table."""
    _check_keys(data, _DATASET_KEYS, "dataset")
    selectors = [key for key in ("limit", "fraction", "ids") if key in data]
    if len(selectors) > 1:
        raise ValueError(f"Use only one of: {', '.join(selectors)}.")
    fraction = data.get("fraction")
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("fraction must be above 0 and at most 1.")
    limit = data.get("limit")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least 1.")
    ids = data.get("ids")
    return DatasetEntry(
        name=data["name"],
        split=data["split"],
        input=artifact_kind(data["input"]),
        metrics=tuple(data["metrics"]),
        limit=limit,
        fraction=fraction,
        ids=tuple(ids) if ids is not None else None,
        seed=data.get("seed", 42),
    )


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Read and check an experiment file.

    Args:
        path: TOML file.

    Returns:
        The parsed experiment.

    Raises:
        ValueError: If the file is incomplete or contains unknown settings.
    """
    path = Path(path).expanduser().resolve(strict=True)
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    try:
        _check_keys(data, _EXPERIMENT_KEYS, "experiment")
        datasets = tuple(_dataset_entry(entry) for entry in data["datasets"])
        settings = dict(data["pipeline"])
        pipeline_type = settings.pop("type")
        name = data["name"]
    except KeyError as error:
        raise ValueError(f"Missing setting in {path}: {error}") from error
    names = [entry.name for entry in datasets]
    if not names:
        raise ValueError("An experiment needs at least one dataset.")
    if len(names) != len(set(names)):
        raise ValueError("Each dataset can appear only once in an experiment.")
    return ExperimentConfig(name, pipeline_type, settings, datasets)
