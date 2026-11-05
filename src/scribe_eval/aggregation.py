"""Combine per-sample scores into a run's results."""

from collections import defaultdict
from statistics import fmean

from .storage import RunPaths, latest_rows, read_jsonl
from .types import Score
from .validation import DatasetPlan


def _dataset_results(paths: RunPaths, plan: DatasetPlan) -> dict[str, object]:
    """Summarize one dataset: coverage first, then each metric's aggregates."""
    name = plan.entry.name
    predictions = latest_rows(read_jsonl(paths.predictions(name)))
    failures = [row for row in predictions if row["status"] != "success"]

    grouped: dict[str, list[Score]] = defaultdict(list)
    for row in read_jsonl(paths.scores(name)):
        grouped[row["metric"]].append(
            Score(row["metric"], row["value"], row["details"])
        )

    metrics = {}
    for metric in plan.metrics:
        scores = grouped.get(metric.name, [])
        if scores:
            metrics[metric.name] = {"n": len(scores), **metric.aggregate(scores)}

    return {
        "n_selected": len(plan.sample_ids),
        "n_run": len(predictions),
        "n_failed": len(failures),
        "failures": {row["sample_id"]: row["error"] for row in failures},
        "metrics": metrics,
        "not_applicable": plan.not_applicable,
    }


def _overall(datasets: dict[str, dict]) -> dict[str, dict[str, float]]:
    """Average, unweighted, the metrics that appear in at least two datasets."""
    collected: dict[str, list[dict]] = defaultdict(list)
    for result in datasets.values():
        for metric, values in result["metrics"].items():
            collected[metric].append(values)

    overall = {}
    for metric, entries in collected.items():
        if len(entries) < 2:
            continue
        keys = set.intersection(*(set(entry) for entry in entries)) - {"n"}
        overall[metric] = {key: fmean(entry[key] for entry in entries) for key in keys}
    return overall


def aggregate_run(paths: RunPaths, plans: tuple[DatasetPlan, ...]) -> dict[str, object]:
    """Build the results document; `overall` is a macro average across datasets."""
    datasets = {plan.entry.name: _dataset_results(paths, plan) for plan in plans}
    return {"datasets": datasets, "overall": _overall(datasets)}
