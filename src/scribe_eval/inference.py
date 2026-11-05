"""First phase: pass samples through the pipeline and record what came out."""

from collections.abc import Callable
from dataclasses import replace
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
    pipeline: Pipeline, sample: Sample, plan: DatasetPlan, paths: RunPaths
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
    )


def run_inference(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    pipeline: Pipeline,
    *,
    retry_errors: bool = False,
    on_prediction: ProgressCallback | None = None,
) -> None:
    """Run every pending sample, appending one row each.

    Resources stay loaded across datasets. Safe to call again on the same run:
    samples with a recorded result are skipped.

    Args:
        paths: Run layout.
        plans: Validated datasets.
        pipeline: Pipeline under evaluation.
        retry_errors: Run failed samples again; the new row supersedes the old one.
        on_prediction: Called after each sample with the prediction, its position,
            and the number of pending samples.
    """
    with pipeline.open(paths.next_telemetry()):
        for plan in plans:
            pending = _pending_ids(paths, plan, retry_errors=retry_errors)
            if not pending:
                continue
            startup = pipeline.prepare(plan.entry.input)
            append_jsonl(paths.startup, {"dataset": plan.entry.name, **startup})
            for position, sample_id in enumerate(pending, start=1):
                sample = plan.dataset.sample(sample_id)
                prediction = _predict(pipeline, sample, plan, paths)
                save_artifacts(prediction, paths.root)
                append_jsonl(
                    paths.predictions(plan.entry.name), prediction_row(prediction)
                )
                if on_prediction is not None:
                    on_prediction(prediction, position, len(pending))
