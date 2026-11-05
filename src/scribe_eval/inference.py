"""First phase: pass samples through the pipeline and record what came out."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from .pipelines import Pipeline
from .storage import (
    RunPaths,
    append_jsonl,
    latest_rows,
    prediction_row,
    read_jsonl,
    save_artifacts,
)
from .types import Prediction, Sample
from .validation import DatasetPlan

ProgressCallback = Callable[[Prediction, int, int], None]

CLEAN_VARIANT = "clean"
FIRST_REPEAT = 0


def _pending_ids(
    paths: RunPaths, plan: DatasetPlan, *, retry_errors: bool
) -> list[str]:
    """List samples without a recorded result; errors count as results by default."""
    rows = latest_rows(read_jsonl(paths.predictions(plan.entry.name)))
    finished = {
        row["sample_id"]
        for row in rows
        if row["status"] == "success" or not retry_errors
    }
    return [sample_id for sample_id in plan.sample_ids if sample_id not in finished]


def _predict(
    pipeline: Pipeline,
    sample: Sample,
    plan: DatasetPlan,
    paths: RunPaths,
    telemetry_directory: Path,
) -> Prediction:
    """Run one sample, timing it from the outside."""
    directory = paths.next_attempt(plan.entry.name, sample.id)
    directory.mkdir(parents=True)
    started = perf_counter()
    result = pipeline.run(sample.artifact(plan.entry.input), directory / "pipeline")
    wall_seconds = perf_counter() - started
    return Prediction(
        dataset=plan.entry.name,
        sample_id=sample.id,
        variant=CLEAN_VARIANT,
        repeat=FIRST_REPEAT,
        input_kind=plan.entry.input,
        status=result.status,
        error=result.error,
        artifacts=result.artifacts,
        telemetry=replace(result.telemetry, wall_seconds=wall_seconds),
        run_directory=str(directory.relative_to(paths.root)),
        telemetry_directory=str(telemetry_directory.relative_to(paths.root)),
    )


def _record(
    pipeline: Pipeline,
    sample_id: str,
    plan: DatasetPlan,
    paths: RunPaths,
    telemetry_directory: Path,
) -> Prediction:
    """Run one sample and save it before moving on."""
    sample = plan.dataset.sample(sample_id)
    prediction = _predict(pipeline, sample, plan, paths, telemetry_directory)
    save_artifacts(prediction, paths.root)
    append_jsonl(paths.predictions(plan.entry.name), prediction_row(prediction))
    return prediction


def _prepare(pipeline: Pipeline, plan: DatasetPlan, paths: RunPaths) -> None:
    """Load what a dataset's input needs, keeping load time apart from inference."""
    startup = pipeline.prepare(plan.entry.input)
    append_jsonl(paths.startup, {"dataset": plan.entry.name, **startup})


def _run_isolated(
    paths: RunPaths,
    plan: DatasetPlan,
    pending: list[str],
    pipeline: Pipeline,
    on_prediction: ProgressCallback | None,
) -> None:
    """Open the pipeline afresh for each sample, so none can affect the next."""
    for position, sample_id in enumerate(pending, start=1):
        telemetry_directory = paths.next_telemetry()
        with pipeline.open(telemetry_directory):
            _prepare(pipeline, plan, paths)
            prediction = _record(pipeline, sample_id, plan, paths, telemetry_directory)
        if on_prediction is not None:
            on_prediction(prediction, position, len(pending))


def _run_shared(
    paths: RunPaths,
    plan: DatasetPlan,
    pending: list[str],
    pipeline: Pipeline,
    telemetry_directory: Path,
    on_prediction: ProgressCallback | None,
) -> None:
    """Run samples in an open pipeline, as a long-lived deployment would."""
    _prepare(pipeline, plan, paths)
    for position, sample_id in enumerate(pending, start=1):
        prediction = _record(pipeline, sample_id, plan, paths, telemetry_directory)
        if on_prediction is not None:
            on_prediction(prediction, position, len(pending))


def run_inference(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    pipeline: Pipeline,
    *,
    isolate_samples: bool = False,
    retry_errors: bool = False,
    on_prediction: ProgressCallback | None = None,
) -> None:
    """Run every pending sample, appending one row each.

    Safe to call again on the same run: samples with a recorded result are skipped.

    Args:
        paths: Run layout.
        plans: Validated datasets.
        pipeline: Pipeline under evaluation.
        isolate_samples: Reopen the pipeline for each sample, so caches and memory
            can't carry over. By default it stays open, as in a deployment.
        retry_errors: Run failed samples again; the new row supersedes the old one.
        on_prediction: Called after each sample with the prediction, its position,
            and the number of pending samples.
    """
    work = [
        (plan, _pending_ids(paths, plan, retry_errors=retry_errors)) for plan in plans
    ]
    work = [(plan, pending) for plan, pending in work if pending]
    if isolate_samples:
        for plan, pending in work:
            _run_isolated(paths, plan, pending, pipeline, on_prediction)
        return
    telemetry_directory = paths.next_telemetry()
    with pipeline.open(telemetry_directory):
        for plan, pending in work:
            _run_shared(
                paths, plan, pending, pipeline, telemetry_directory, on_prediction
            )
