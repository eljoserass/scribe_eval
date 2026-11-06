"""Entry points shared by the CLI and Python callers."""

import platform
import shutil
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .aggregation import aggregate_run
from .config import ExperimentConfig, load_experiment
from .datasets import CONVERTERS
from .datasets.prepared import PreparedDataset
from .hardware import describe_hardware
from .inference import ProgressCallback, run_inference
from .judge import Asker, Judge
from .pipelines import Pipeline, create_pipeline
from .reporting import write_report
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


def _pipeline_hash(pipeline: Pipeline) -> str:
    """Fingerprint a pipeline's effective configuration."""
    return sha256_text(repr(sorted(pipeline.describe().items())))


def _judge(
    experiment: ExperimentConfig, paths: RunPaths, judge: Asker | None
) -> Asker | None:
    """Use a given judge, or build the configured one with the run's cache."""
    if judge is not None or experiment.judge is None:
        return judge
    return Judge(experiment.judge, paths.judge_cache)


def _plans(
    experiment: ExperimentConfig,
    pipeline: Pipeline,
    data_directory: Path,
    judge: Asker | None,
    frozen: dict[str, list[str]] | None = None,
) -> tuple[DatasetPlan, ...]:
    """Validate every dataset, reusing frozen sample IDs when given."""
    plans = []
    for entry in experiment.datasets:
        dataset = PreparedDataset(data_directory / entry.name)
        ids = frozen[entry.name] if frozen else None
        plans.append(
            plan_dataset(entry, dataset, pipeline, sample_ids=ids, judge=judge)
        )
    return tuple(plans)


def _package_version(name: str) -> str | None:
    """Read an installed package's version."""
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _manifest(
    experiment: ExperimentConfig,
    pipeline: Pipeline,
    plans: tuple[DatasetPlan, ...],
    *,
    text: str,
    base_directory: Path,
) -> dict[str, object]:
    """Record what a run needs for later interpretation and resuming."""
    return {
        "name": experiment.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_sha256": sha256_text(text),
        "experiment_directory": str(base_directory),
        "pipeline_type": experiment.pipeline_type,
        "isolate_samples": experiment.isolate_samples,
        "pipeline_sha256": _pipeline_hash(pipeline),
        "pipeline": pipeline.describe(),
        "judge": experiment.judge.model if experiment.judge else None,
        "hardware": describe_hardware(),
        "versions": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scribe_eval": _package_version("scribe-eval"),
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


def _score(paths: RunPaths, plans: tuple[DatasetPlan, ...]) -> dict[str, object]:
    """Score saved predictions and write the results."""
    for plan in plans:
        score_dataset(paths, plan)
    results = aggregate_run(paths, plans)
    write_json(paths.results, results)
    return results


def _execute(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    pipeline: Pipeline,
    *,
    isolate_samples: bool,
    retry_errors: bool,
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
            pipeline,
            isolate_samples=isolate_samples,
            retry_errors=retry_errors,
            on_prediction=track,
        )
        _write_status(paths, plans, "scoring")
        results = _score(paths, plans)
        _write_status(paths, plans, "completed")
        write_report(paths, plans, results, pipeline)
        return results
    except KeyboardInterrupt:
        _write_status(paths, plans, "interrupted")
        raise
    except Exception as error:
        _write_status(paths, plans, "failed", str(error))
        raise


def run_experiment(
    experiment_path: str | Path,
    *,
    data_directory: Path = Path("data"),
    runs_directory: Path = Path("runs"),
    run_directory: Path | None = None,
    pipeline: Pipeline | None = None,
    judge: Asker | None = None,
    on_prediction: ProgressCallback | None = None,
) -> tuple[Path, dict[str, object]]:
    """Validate an experiment, then run and score it.

    Args:
        experiment_path: TOML definition.
        data_directory: Root of prepared datasets.
        runs_directory: Where a new run directory is created.
        run_directory: Exact directory for this run, which must not exist.
        pipeline: Replaces the configured pipeline, for tests.
        judge: Replaces the configured judge, for tests.
        on_prediction: Called after each sample.

    Returns:
        The run directory and its results.

    Raises:
        ValueError: If the experiment is invalid. Nothing is written in that case.
    """
    experiment_path = Path(experiment_path).expanduser().resolve(strict=True)
    experiment = load_experiment(experiment_path)
    base_directory = experiment_path.parent
    pipeline = pipeline or create_pipeline(
        experiment.pipeline_type,
        experiment.pipeline_settings,
        base_directory=base_directory,
    )
    root = run_directory or new_run_directory(runs_directory, experiment.name)
    root = root.expanduser().resolve()
    if root.exists():
        raise ValueError(f"Run directory already exists: {root}")
    paths = RunPaths(root)
    plans = _plans(
        experiment, pipeline, data_directory, _judge(experiment, paths, judge)
    )
    root.mkdir(parents=True)
    shutil.copyfile(experiment_path, paths.experiment)
    text = experiment_path.read_text(encoding="utf-8")
    manifest = _manifest(
        experiment, pipeline, plans, text=text, base_directory=base_directory
    )
    write_json(paths.manifest, manifest)

    results = _execute(
        paths,
        plans,
        pipeline,
        isolate_samples=experiment.isolate_samples,
        retry_errors=False,
        on_prediction=on_prediction,
    )
    return root, results


def _load_run(
    run_directory: Path,
    data_directory: Path,
    pipeline: Pipeline | None,
    judge: Asker | None,
) -> tuple[RunPaths, tuple[DatasetPlan, ...], Pipeline]:
    """Rebuild a run's plans, refusing data or configuration that changed."""
    paths = RunPaths(Path(run_directory).expanduser().resolve(strict=True))
    manifest = read_json(paths.manifest)
    experiment = load_experiment(paths.experiment)
    pipeline = pipeline or create_pipeline(
        experiment.pipeline_type,
        experiment.pipeline_settings,
        base_directory=Path(manifest["experiment_directory"]),
    )
    if _pipeline_hash(pipeline) != manifest["pipeline_sha256"]:
        raise ValueError("The pipeline changed since this run started.")

    recorded = manifest["datasets"]
    frozen = {name: data["sample_ids"] for name, data in recorded.items()}
    judge = _judge(experiment, paths, judge)
    plans = _plans(experiment, pipeline, data_directory, judge, frozen)
    for plan in plans:
        if plan.dataset.version != recorded[plan.entry.name]["version"]:
            raise ValueError(f"Dataset {plan.entry.name!r} changed version.")
    return paths, plans, pipeline


def resume_run(
    run_directory: Path,
    *,
    data_directory: Path = Path("data"),
    retry_errors: bool = False,
    pipeline: Pipeline | None = None,
    judge: Asker | None = None,
    on_prediction: ProgressCallback | None = None,
) -> dict[str, object]:
    """Finish a run with the same samples and configuration.

    Args:
        run_directory: Existing run.
        data_directory: Root of prepared datasets.
        retry_errors: Run failed samples again.
        pipeline: Replaces the configured pipeline, for tests.
        judge: Replaces the configured judge, for tests.
        on_prediction: Called after each sample.

    Returns:
        The run's results.

    Raises:
        ValueError: If the pipeline or a dataset changed.
    """
    paths, plans, pipeline = _load_run(run_directory, data_directory, pipeline, judge)
    return _execute(
        paths,
        plans,
        pipeline,
        isolate_samples=load_experiment(paths.experiment).isolate_samples,
        retry_errors=retry_errors,
        on_prediction=on_prediction,
    )


def score_run(
    run_directory: Path,
    *,
    data_directory: Path = Path("data"),
    pipeline: Pipeline | None = None,
    judge: Asker | None = None,
) -> dict[str, object]:
    """Recompute scores, results, and the report, without running the pipeline."""
    paths, plans, pipeline = _load_run(run_directory, data_directory, pipeline, judge)
    results = _score(paths, plans)
    _write_status(paths, plans, "completed")
    write_report(paths, plans, results, pipeline)
    return results
