"""Check that a dataset, a workflow, and metrics fit before loading any model."""

from collections.abc import Sequence
from dataclasses import dataclass

from scribnic import ArtifactKind, WorkflowConfig
from scribnic.planning import resolve_plan

from .config import DatasetEntry
from .datasets.prepared import PreparedDataset
from .metrics import resolve_metrics
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


def produced_kinds(
    workflow: WorkflowConfig, input_kind: ArtifactKind
) -> set[ArtifactKind]:
    """List what a workflow produces from an input, excluding the input itself.

    Raises:
        ValueError: If the workflow does not accept the input.
    """
    plan = resolve_plan(workflow, input_kind, workflow.default_output)
    return {planned.step.model.output for planned in plan}


def plan_dataset(
    entry: DatasetEntry,
    dataset: PreparedDataset,
    workflow: WorkflowConfig,
    *,
    sample_ids: Sequence[str] | None = None,
) -> DatasetPlan:
    """Validate one dataset entry against its data and workflow.

    Args:
        entry: The experiment's dataset settings.
        dataset: Prepared data on disk.
        workflow: Workflow under evaluation.
        sample_ids: Frozen IDs when resuming; selected from `entry` otherwise.

    Returns:
        The plan. A metric the workflow cannot satisfy is listed as not applicable.

    Raises:
        ValueError: If the dataset lacks the input or a reference a metric needs.
    """
    if entry.split not in dataset.splits:
        raise ValueError(f"Dataset {dataset.name!r} has no split {entry.split!r}.")
    if not satisfies(dataset.provides, entry.input):
        raise ValueError(f"Dataset {dataset.name!r} has no {entry.input!r} input.")
    produced = produced_kinds(workflow, entry.input)

    metrics = []
    not_applicable = {}
    for metric in resolve_metrics(entry.metrics):
        for kind in metric.requires.reference:
            if not satisfies(dataset.provides, kind):
                raise ValueError(
                    f"Metric {metric.name!r} needs a {kind!r} reference, "
                    f"which dataset {dataset.name!r} does not have."
                )
        missing = [k for k in metric.requires.predicted if not satisfies(produced, k)]
        if missing:
            not_applicable[metric.name] = (
                f"{workflow.name} does not produce {', '.join(missing)} "
                f"from {entry.input}"
            )
        else:
            metrics.append(metric)

    if sample_ids is None:
        sample_ids = select_ids(entry, dataset.splits[entry.split])
    return DatasetPlan(
        entry, dataset, tuple(sample_ids), tuple(metrics), not_applicable
    )
