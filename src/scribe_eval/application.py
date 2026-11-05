"""Entry points shared by the CLI and Python callers."""

import platform
import shutil
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from scribnic import WorkflowConfig, load_workflow
from scribnic.planning import validate_workflow
from scribnic.workflow import BackendFactory

from .aggregation import aggregate_run
from .config import ExperimentConfig, load_experiment
from .datasets import CONVERTERS
from .datasets.prepared import PreparedDataset
from .inference import ProgressCallback, run_inference
from .scoring import score_dataset
from .storage import (
    RunPaths,
    latest_rows,
    new_run_directory,
    read_json,
    read_jsonl,
    sha256_text,
    write_json,
)
from .validation import DatasetPlan, plan_dataset


def prepare_dataset(
    name: str,
    data_directory: Path,
    *,
    source: Path | None = None,
    ids: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> PreparedDataset:
    """Convert a raw dataset to the common format.

    Args:
        name: Registered converter.
        data_directory: Root of prepared datasets.
        source: Raw data location; each converter defines its default.
        ids: Convert only these samples.
        limit: Convert only the first samples, to try a dataset cheaply.

    Returns:
        The prepared dataset, including samples converted earlier.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in CONVERTERS:
        raise ValueError(f"Unknown dataset {name!r}: {', '.join(sorted(CONVERTERS))}")
    output = data_directory / name
    CONVERTERS[name](output, source=source, ids=ids, limit=limit)
    return PreparedDataset(output)


def _workflow_hash(workflow: WorkflowConfig) -> str:
    """Fingerprint a workflow's effective configuration."""
    return sha256_text(repr(sorted(workflow.to_dict().items())))


def _plans(
    experiment: ExperimentConfig,
    workflow: WorkflowConfig,
    data_directory: Path,
    frozen: dict[str, list[str]] | None = None,
) -> tuple[DatasetPlan, ...]:
    """Validate every dataset, reusing frozen sample IDs when given."""
    plans = []
    for entry in experiment.datasets:
        dataset = PreparedDataset(data_directory / entry.name)
        ids = frozen[entry.name] if frozen else None
        plans.append(plan_dataset(entry, dataset, workflow, sample_ids=ids))
    return tuple(plans)


def _manifest(
    experiment: ExperimentConfig,
    workflow: WorkflowConfig,
    plans: tuple[DatasetPlan, ...],
    text: str,
) -> dict[str, object]:
    """Record what a run needs for later interpretation and resuming."""
    return {
        "name": experiment.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_sha256": sha256_text(text),
        "workflow_source": experiment.workflow,
        "workflow_sha256": _workflow_hash(workflow),
        "workflow": workflow.to_dict(),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scribe_eval": version("scribe-eval"),
            "scribnic": version("scribnic"),
        },
        "datasets": {
            plan.entry.name: {
                "version": plan.dataset.version,
                "language": plan.dataset.language,
                "license": plan.dataset.license,
                "split": plan.entry.split,
                "input": plan.entry.input,
                "sample_ids": list(plan.sample_ids),
                "metrics": {metric.name: metric.version for metric in plan.metrics},
                "not_applicable": plan.not_applicable,
            }
            for plan in plans
        },
    }


def _write_status(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    state: str,
    error: str | None = None,
) -> None:
    """Summarize progress from the JSONL files, which remain the source of truth."""
    datasets = {}
    for plan in plans:
        rows = latest_rows(read_jsonl(paths.predictions(plan.entry.name)))
        failed = sum(row["status"] != "success" for row in rows)
        datasets[plan.entry.name] = {
            "selected": len(plan.sample_ids),
            "run": len(rows),
            "failed": failed,
        }
    write_json(
        paths.status,
        {
            "state": state,
            "error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "datasets": datasets,
        },
    )


def _score_and_aggregate(
    paths: RunPaths, plans: tuple[DatasetPlan, ...]
) -> dict[str, object]:
    """Score saved predictions and write the results."""
    for plan in plans:
        score_dataset(paths, plan)
    results = aggregate_run(paths, plans)
    write_json(paths.results, results)
    return results


def _execute(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    workflow: WorkflowConfig,
    *,
    retry_errors: bool,
    download: bool | None,
    backend_factory: BackendFactory | None,
    on_prediction: ProgressCallback | None,
) -> dict[str, object]:
    """Run inference and scoring, keeping status truthful on every exit path."""

    def track(prediction, position, total) -> None:
        """Refresh the status, then report to the caller."""
        _write_status(paths, plans, "running")
        if on_prediction is not None:
            on_prediction(prediction, position, total)

    _write_status(paths, plans, "running")
    try:
        run_inference(
            paths,
            plans,
            workflow,
            retry_errors=retry_errors,
            download=download,
            backend_factory=backend_factory,
            on_prediction=track,
        )
        results = _score_and_aggregate(paths, plans)
    except KeyboardInterrupt:
        _write_status(paths, plans, "interrupted")
        raise
    except Exception as error:
        _write_status(paths, plans, "failed", str(error))
        raise
    _write_status(paths, plans, "completed")
    return results


def run_experiment(
    experiment_path: str | Path,
    *,
    data_directory: Path = Path("data"),
    runs_directory: Path = Path("runs"),
    run_directory: Path | None = None,
    workflow: WorkflowConfig | None = None,
    download: bool | None = None,
    backend_factory: BackendFactory | None = None,
    on_prediction: ProgressCallback | None = None,
) -> tuple[Path, dict[str, object]]:
    """Validate an experiment, then run and score it.

    Args:
        experiment_path: TOML definition.
        data_directory: Root of prepared datasets.
        runs_directory: Where a new run directory is created.
        run_directory: Exact directory for this run, which must not exist.
        workflow: Parsed workflow overriding the experiment's, for tests.
        download: Passed to Scribnic; False requires cached models.
        backend_factory: Replaces model backends, for tests.
        on_prediction: Called after each sample.

    Returns:
        The run directory and its results.

    Raises:
        ValueError: If the experiment is invalid. Nothing is written in that case.
    """
    experiment_path = Path(experiment_path).expanduser().resolve(strict=True)
    experiment = load_experiment(experiment_path)
    workflow = workflow or load_workflow(experiment.workflow)
    validate_workflow(workflow)
    plans = _plans(experiment, workflow, data_directory)

    root = run_directory or new_run_directory(runs_directory, experiment.name)
    root = root.expanduser().resolve()
    if root.exists():
        raise ValueError(f"Run directory already exists: {root}")
    root.mkdir(parents=True)
    paths = RunPaths(root)
    shutil.copyfile(experiment_path, paths.experiment)
    text = experiment_path.read_text(encoding="utf-8")
    write_json(paths.manifest, _manifest(experiment, workflow, plans, text))

    results = _execute(
        paths,
        plans,
        workflow,
        retry_errors=False,
        download=download,
        backend_factory=backend_factory,
        on_prediction=on_prediction,
    )
    return root, results


def _load_run(
    run_directory: Path, data_directory: Path, workflow: WorkflowConfig | None
) -> tuple[RunPaths, tuple[DatasetPlan, ...], WorkflowConfig]:
    """Rebuild a run's plans, refusing data or configuration that changed."""
    paths = RunPaths(Path(run_directory).expanduser().resolve(strict=True))
    manifest = read_json(paths.manifest)
    experiment = load_experiment(paths.experiment)
    experiment = replace(experiment, workflow=manifest["workflow_source"])
    workflow = workflow or load_workflow(experiment.workflow)
    if _workflow_hash(workflow) != manifest["workflow_sha256"]:
        raise ValueError("The workflow changed since this run started.")

    recorded = manifest["datasets"]
    frozen = {name: data["sample_ids"] for name, data in recorded.items()}
    plans = _plans(experiment, workflow, data_directory, frozen)
    for plan in plans:
        if plan.dataset.version != recorded[plan.entry.name]["version"]:
            raise ValueError(f"Dataset {plan.entry.name!r} changed version.")
    return paths, plans, workflow


def resume_run(
    run_directory: Path,
    *,
    data_directory: Path = Path("data"),
    retry_errors: bool = False,
    workflow: WorkflowConfig | None = None,
    download: bool | None = None,
    backend_factory: BackendFactory | None = None,
    on_prediction: ProgressCallback | None = None,
) -> dict[str, object]:
    """Finish a run with the same samples and configuration.

    Args:
        run_directory: Existing run.
        data_directory: Root of prepared datasets.
        retry_errors: Run failed samples again.
        workflow: Parsed workflow overriding the recorded one, for tests.
        download: Passed to Scribnic; False requires cached models.
        backend_factory: Replaces model backends, for tests.
        on_prediction: Called after each sample.

    Returns:
        The run's results.

    Raises:
        ValueError: If the workflow or a dataset changed.
    """
    paths, plans, workflow = _load_run(run_directory, data_directory, workflow)
    return _execute(
        paths,
        plans,
        workflow,
        retry_errors=retry_errors,
        download=download,
        backend_factory=backend_factory,
        on_prediction=on_prediction,
    )


def score_run(
    run_directory: Path,
    *,
    data_directory: Path = Path("data"),
    workflow: WorkflowConfig | None = None,
) -> dict[str, object]:
    """Recompute scores and results from saved predictions, without any model."""
    paths, plans, _ = _load_run(run_directory, data_directory, workflow)
    return _score_and_aggregate(paths, plans)
