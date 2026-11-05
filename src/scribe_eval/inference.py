"""First phase: pass samples through the workflow and record what came out."""

from collections.abc import Callable
from pathlib import Path

from scribnic import RunResult, WorkflowConfig, WorkflowRunError, open_workflow
from scribnic.workflow import BackendFactory, WorkflowSession

from .storage import (
    RunPaths,
    append_jsonl,
    latest_rows,
    prediction_row,
    read_jsonl,
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


def _step_summary(result: RunResult) -> tuple[dict[str, object], ...]:
    """Keep the step data metrics need; the full record stays in Scribnic's run."""
    return tuple(
        {
            "name": step.name,
            "backend": step.backend,
            "elapsed_seconds": step.elapsed_seconds,
            "usage": step.metadata.get("usage"),
        }
        for step in result.steps
    )


def _prediction(
    result: RunResult, sample: Sample, plan: DatasetPlan, directory: Path, root: Path
) -> Prediction:
    """Convert a Scribnic result, leaving out the input artifact it also carries."""
    produced = {
        artifact.kind: artifact
        for artifact in result.artifacts
        if artifact.kind != result.source.kind
    }
    steps = _step_summary(result)
    return Prediction(
        dataset=plan.entry.name,
        sample_id=sample.id,
        variant=CLEAN_VARIANT,
        repeat=FIRST_REPEAT,
        input_kind=plan.entry.input,
        status="success" if result.status == "success" else "error",
        error=result.error,
        artifacts=produced,
        steps=steps,
        resources=result.resources,
        elapsed_seconds=sum(float(step["elapsed_seconds"]) for step in steps),
        run_directory=str(directory.relative_to(root)),
    )


def _predict(
    session: WorkflowSession, sample: Sample, plan: DatasetPlan, paths: RunPaths
) -> Prediction:
    """Run one sample; a failed execution is a result, not an exception."""
    directory = paths.next_attempt(plan.entry.name, sample.id)
    source = sample.artifact(plan.entry.input)
    try:
        result = session.run(source, output_dir=directory)
    except WorkflowRunError as error:
        result = error.result
    return _prediction(result, sample, plan, directory, paths.root)


def run_inference(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    workflow: WorkflowConfig,
    *,
    retry_errors: bool = False,
    download: bool | None = None,
    backend_factory: BackendFactory | None = None,
    on_prediction: ProgressCallback | None = None,
) -> None:
    """Run every pending sample, appending one row each.

    Models load once and stay loaded across datasets. Safe to call again on the same
    run: samples with a recorded result are skipped.

    Args:
        paths: Run layout.
        plans: Validated datasets.
        workflow: Workflow under evaluation.
        retry_errors: Run failed samples again; the new row supersedes the old one.
        download: Passed to Scribnic; False requires cached models.
        backend_factory: Replaces model backends, for tests.
        on_prediction: Called after each sample with the prediction, its position,
            and the number of pending samples.
    """
    with open_workflow(
        workflow, backend_factory=backend_factory, download=download
    ) as session:
        for plan in plans:
            pending = _pending_ids(paths, plan, retry_errors=retry_errors)
            if not pending:
                continue
            startup = session.start(source_kind=plan.entry.input)
            append_jsonl(paths.startup, {"dataset": plan.entry.name, **startup})
            for position, sample_id in enumerate(pending, start=1):
                sample = plan.dataset.sample(sample_id)
                prediction = _predict(session, sample, plan, paths)
                append_jsonl(
                    paths.predictions(plan.entry.name), prediction_row(prediction)
                )
                if on_prediction is not None:
                    on_prediction(prediction, position, len(pending))
