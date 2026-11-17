"""Compare finished runs, metric by metric, from their saved records.

One experiment evaluates one pipeline, so pipelines are compared through their runs.
Nothing here runs a model or rescores: it reads what each run aggregated, and warns
when the runs are not comparable as recorded.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

Coverage = dict[str, int]
Statistic = tuple[str, str]


@dataclass(frozen=True)
class RunRecord:
    """What a finished run left on disk, already parsed. Holds no resources.

    Attributes:
        label: Short name for the run, usually its directory name.
        manifest: Contents of `manifest.json`.
        status: Contents of `status.json`.
        results: Contents of `results.json`.
    """

    label: str
    manifest: dict
    status: dict
    results: dict


@dataclass(frozen=True)
class DatasetComparison:
    """One dataset across runs; a run without the dataset has no entries.

    Attributes:
        coverage: `n_selected`, `n_run`, and `n_failed` by run label.
        values: By `(metric, statistic)`, each run's value.
        not_applicable: By metric, each run's reason it was ruled out.
    """

    coverage: dict[str, Coverage] = field(default_factory=dict)
    values: dict[Statistic, dict[str, float]] = field(default_factory=dict)
    not_applicable: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Comparison:
    """Runs side by side, with reasons to distrust the comparison.

    Attributes:
        runs: The compared runs, in the order given.
        datasets: By dataset name, in first-seen order.
        warnings: Differences that make values not directly comparable.
    """

    runs: tuple[RunRecord, ...]
    datasets: dict[str, DatasetComparison]
    warnings: tuple[str, ...]


def compare(runs: Sequence[RunRecord]) -> Comparison:
    """Line up each run's aggregates by dataset, metric, and statistic.

    Args:
        runs: At least two runs with distinct labels.

    Returns:
        The comparison. Values are copied as recorded; nothing is recomputed.

    Raises:
        ValueError: If fewer than two runs are given, or labels repeat.
    """
    if len(runs) < 2:
        raise ValueError("Comparing needs at least two runs.")
    labels = [run.label for run in runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f"Run labels repeat: {', '.join(labels)}")

    datasets: dict[str, DatasetComparison] = {}
    for run in runs:
        for name, result in run.results["datasets"].items():
            entry = datasets.setdefault(name, DatasetComparison())
            _add_run(entry, run.label, result)
    return Comparison(tuple(runs), datasets, tuple(_warnings(runs)))


def _add_run(entry: DatasetComparison, label: str, result: dict) -> None:
    """Copy one run's results for one dataset into the comparison."""
    entry.coverage[label] = {
        key: result[key] for key in ("n_selected", "n_run", "n_failed")
    }
    for metric, aggregates in result["metrics"].items():
        for statistic, value in aggregates.items():
            if statistic == "n":
                continue
            entry.values.setdefault((metric, statistic), {})[label] = value
    for metric, reason in result["not_applicable"].items():
        entry.not_applicable.setdefault(metric, {})[label] = reason


def _warnings(runs: Sequence[RunRecord]) -> list[str]:
    """List what differs between runs in ways that bias a comparison.

    Runs that are unfinished, scored different samples or data versions, or used a
    different version of a metric can still be shown, but not read as like for like.
    """
    warnings = []
    for run in runs:
        state = run.status.get("state")
        if state != "completed":
            warnings.append(f"{run.label} is {state}, not completed.")

    for name in _shared_datasets(runs):
        recorded = [run.manifest["datasets"][name] for run in runs]
        if len({tuple(data["sample_ids"]) for data in recorded}) > 1:
            warnings.append(f"{name}: runs scored different samples.")
        if len({data["version"] for data in recorded}) > 1:
            warnings.append(f"{name}: runs used different dataset versions.")
        if len({data["input"] for data in recorded}) > 1:
            warnings.append(f"{name}: runs fed the pipeline different inputs.")
        for metric in _metrics_with_several_versions(recorded):
            warnings.append(f"{name}: metric {metric!r} changed version.")
    return warnings


def _shared_datasets(runs: Sequence[RunRecord]) -> list[str]:
    """Datasets every run evaluated, in the first run's order."""
    shared = list(runs[0].manifest["datasets"])
    for run in runs[1:]:
        shared = [name for name in shared if name in run.manifest["datasets"]]
    return shared


def _metrics_with_several_versions(recorded: Sequence[dict]) -> list[str]:
    """Metrics recorded with more than one version across runs.

    Older manifests lack metric versions; they cannot be checked and are skipped.
    """
    versions: dict[str, set[int]] = {}
    for data in recorded:
        for metric, version in data.get("metrics", {}).items():
            versions.setdefault(metric, set()).add(version)
    return sorted(metric for metric, seen in versions.items() if len(seen) > 1)
