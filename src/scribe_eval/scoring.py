"""Second phase: score saved predictions, without running any model."""

from .storage import RunPaths, latest_rows, load_prediction, read_jsonl, write_jsonl
from .types import Metric, Prediction, find_artifact
from .validation import DatasetPlan


def _can_score(metric: Metric, prediction: Prediction) -> bool:
    """Tell whether a prediction has what a metric needs.

    A failed run keeps its partial artifacts, so WER still counts when only the note
    was rejected. Metrics that need no artifact describe the run, and only apply to
    successful ones.
    """
    predicted = metric.requires.predicted
    if not predicted:
        return prediction.status == "success"
    return all(find_artifact(prediction.artifacts, kind) for kind in predicted)


def score_dataset(paths: RunPaths, plan: DatasetPlan) -> None:
    """Score one dataset's predictions, replacing its previous scores."""
    name = plan.entry.name
    rows = []
    for row in latest_rows(read_jsonl(paths.predictions(name))):
        prediction = load_prediction(paths.root, row)
        sample = plan.dataset.sample(prediction.sample_id)
        for metric in plan.metrics:
            if not _can_score(metric, prediction):
                continue
            score = metric.score(prediction, sample)
            if score is None:
                continue
            rows.append(
                {
                    "sample_id": prediction.sample_id,
                    "variant": prediction.variant,
                    "repeat": prediction.repeat,
                    "metric": score.metric,
                    "value": score.value,
                    "details": score.details,
                }
            )
    write_jsonl(paths.scores(name), rows)
