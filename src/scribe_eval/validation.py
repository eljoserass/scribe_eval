"""Check that a dataset, a pipeline, and metrics fit before loading any model."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .config import DatasetEntry
from .datasets.prepared import PreparedDataset
from .metrics import needs_judge, resolve_metrics
from .pipelines import Pipeline
from .selection import select_ids
from .types import Metric, satisfies


@dataclass(frozen=True)
class DatasetPlan:
    """What will run for one dataset, and which metrics were ruled out and why."""

    entry: DatasetEntry
    dataset: PreparedDataset
    sample_ids: tuple[str, ...]
    metrics: tuple[Metric, ...]
    not_applicable: dict[str, str]
    metric_settings: dict[str, dict[str, object]]


def plan_dataset(
    entry: DatasetEntry,
    dataset: PreparedDataset,
    pipeline: Pipeline,
    *,
    sample_ids: Sequence[str] | None = None,
    metrics: Mapping[str, Metric] | None = None,
) -> DatasetPlan:
    """Validate one dataset entry against its data and pipeline.

    Args:
        entry: The experiment's dataset settings.
        dataset: Prepared data on disk.
        pipeline: Pipeline under evaluation.
        sample_ids: Frozen IDs when resuming; selected from `entry` otherwise.
        metrics: Metrics configured for the experiment; registry defaults otherwise.

    Returns:
        The plan. A metric the pipeline cannot satisfy, or one that declares
        `languages` without the dataset's, is listed as not applicable.

    Raises:
        ValueError: If the dataset lacks the input or a reference a metric needs,
            the pipeline doesn't accept the input, a metric needs a missing judge,
            a model metric's extra isn't installed, or a metric doesn't support the
            dataset's language.
    """
    if entry.split not in dataset.splits:
        raise ValueError(f"Dataset {dataset.name!r} has no split {entry.split!r}.")
    if not satisfies(dataset.provides, entry.input):
        raise ValueError(f"Dataset {dataset.name!r} has no {entry.input!r} input.")
    produced = pipeline.produces(entry.input)

    planned = []
    not_applicable = {}
    for metric in resolve_metrics(entry.metrics, metrics):
        if needs_judge(metric) and metric.judge is None:
            raise ValueError(f"Metric {metric.name!r} needs a [judge] table.")
        for kind in metric.requires.reference:
            if not satisfies(dataset.provides, kind):
                raise ValueError(
                    f"Metric {metric.name!r} needs a {kind!r} reference, "
                    f"which dataset {dataset.name!r} does not have."
                )
        missing = [k for k in metric.requires.predicted if not satisfies(produced, k)]
        if missing:
            not_applicable[metric.name] = (
                f"the pipeline does not produce {', '.join(missing)} from {entry.input}"
            )
        elif hasattr(metric, "languages") and dataset.language not in metric.languages:
            not_applicable[metric.name] = (
                f"the metric supports {', '.join(metric.languages)}, "
                f"not {dataset.language}"
            )
        else:
            planned.append(metric)

    if sample_ids is None:
        sample_ids = select_ids(entry, dataset.splits[entry.split])
    # Effective model settings, for the manifest; also checks the extra is there.
    metric_settings = {
        metric.name: metric.describe(dataset.language)
        for metric in planned
        if hasattr(metric, "describe")
    }
    return DatasetPlan(
        entry,
        dataset,
        tuple(sample_ids),
        tuple(planned),
        not_applicable,
        metric_settings,
    )
