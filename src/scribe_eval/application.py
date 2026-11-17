"""Entry points shared by the CLI and Python callers."""

import os
import platform
import shutil
import socket
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .aggregation import aggregate_run
from .comparison import Comparison, RunRecord, compare
from .config import ExperimentConfig, load_experiment
from .datasets import CONVERTERS
from .datasets.prepared import PreparedDataset
from .hardware import describe_hardware
from .inference import ProgressCallback, run_inference
from .judge import Asker, Judge
from .listing import RunFilter, RunSummary, summarize_runs
from .metrics import configure_metrics
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
    metrics = configure_metrics(experiment.metric_settings, judge=judge)
    plans = []
    for entry in experiment.datasets:
        dataset = PreparedDataset(data_directory / entry.name)
        ids = frozen[entry.name] if frozen else None
        plans.append(
            plan_dataset(entry, dataset, pipeline, sample_ids=ids, metrics=metrics)
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
                **_metric_record(plan),
            }
            for plan in plans
        },
    }


def _metric_record(plan: DatasetPlan) -> dict[str, object]:
    """What a dataset is scored with: versions, model settings, and exclusions."""
    return {
        "metrics": {metric.name: metric.version for metric in plan.metrics},
        "metric_settings": plan.metric_settings,
        "not_applicable": plan.not_applicable,
    }


def _record_metrics(paths: RunPaths, plans: tuple[DatasetPlan, ...]) -> None:
    """Update the manifest when re-scoring, since metrics may have changed."""
    manifest = read_json(paths.manifest)
    for plan in plans:
        manifest["datasets"][plan.entry.name].update(_metric_record(plan))
    write_json(paths.manifest, manifest)


def _write_status(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    state: str,
    error: str | None = None,
) -> None:
    """Summarize progress from the JSONL files, which remain the source of truth.

    The process ID and host let `scribe-eval runs` tell a live run from one whose
    process died without updating its state.
    """
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
            "pid": os.getpid(),
            "host": socket.gethostname(),
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
    score: bool = True,
) -> dict[str, object] | None:
    """Run inference, then scoring unless told not to, keeping status truthful.

    Without scoring, the run ends as `predicted`, with no results or report, and
    `score_run` finishes it later.
    """

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
        if not score:
            _write_status(paths, plans, "predicted")
            return None
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
    score: bool = True,
) -> tuple[Path, dict[str, object] | None]:
    """Validate an experiment, then run and score it.

    Args:
        experiment_path: TOML definition.
        data_directory: Root of prepared datasets.
        runs_directory: Where a new run directory is created.
        run_directory: Exact directory for this run, which must not exist.
        pipeline: Replaces the configured pipeline, for tests.
        judge: Replaces the configured judge, for tests.
        on_prediction: Called after each sample.
        score: False saves predictions only, to score later with `score_run`.
            Metrics are still validated first.

    Returns:
        The run directory, and its results, or None when not scored.

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
        score=score,
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
    score: bool = True,
) -> dict[str, object] | None:
    """Finish a run with the same samples and configuration.

    Args:
        run_directory: Existing run.
        data_directory: Root of prepared datasets.
        retry_errors: Run failed samples again.
        pipeline: Replaces the configured pipeline, for tests.
        judge: Replaces the configured judge, for tests.
        on_prediction: Called after each sample.
        score: False saves predictions only, to score later with `score_run`.

    Returns:
        The run's results, or None when not scored.

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
        score=score,
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
    _record_metrics(paths, plans)
    results = _score(paths, plans)
    _write_status(paths, plans, "completed")
    write_report(paths, plans, results, pipeline)
    return results


def compare_runs(run_directories: Sequence[Path]) -> Comparison:
    """Put finished runs side by side, reading only what they saved.

    Args:
        run_directories: Two or more run directories, each with `results.json`.

    Returns:
        Values by dataset, metric, and statistic, with comparability warnings.

    Raises:
        ValueError: If fewer than two runs are given, a run was never scored, or
            two runs share a directory name.
        FileNotFoundError: If a run directory does not exist.
    """
    records = []
    for directory in run_directories:
        paths = RunPaths(Path(directory).expanduser().resolve(strict=True))
        if not paths.results.exists():
            raise ValueError(f"{paths.root.name} has no results; score it first.")
        records.append(
            RunRecord(
                paths.root.name,
                read_json(paths.manifest),
                read_json(paths.status),
                read_json(paths.results),
            )
        )
    return compare(records)


def list_runs(
    runs_directory: Path = Path("runs"), *, which: RunFilter = RunFilter.ALL
) -> list[RunSummary]:
    """List runs with their progress, newest first, reading only saved records.

    Args:
        runs_directory: Parent of run directories.
        which: `running`, `failed`, or `all`; see `summarize_runs`.
    """
    return summarize_runs(runs_directory, which=which)
