"""Build a self-contained HTML report from a run's records."""

import json
from importlib.resources import files
from pathlib import Path

from .pipelines import Pipeline, Timeline
from .storage import RunPaths, latest_rows, load_prediction, read_json, read_jsonl
from .telemetry import MemorySeries
from .telemetry.otlp import execution_directories
from .types import Prediction, Sample, find_artifact
from .validation import DatasetPlan

_MAX_POINTS = 240


def _downsample(series: MemorySeries) -> dict[str, object]:
    """Thin a series for the page, always keeping the peak."""
    stride = max(1, len(series.seconds) // _MAX_POINTS)
    keep = set(range(0, len(series.seconds), stride))
    keep.add(series.bytes.index(max(series.bytes)))
    indices = sorted(keep)
    return {
        "process": series.process,
        "seconds": [round(series.seconds[index], 2) for index in indices],
        "bytes": [series.bytes[index] for index in indices],
    }


def _text(artifacts: dict, kind: str) -> str | None:
    """Read an artifact's text, if it exists."""
    artifact = find_artifact(artifacts, kind)
    return None if artifact is None else artifact.text


def _sample_view(
    prediction: Prediction,
    sample: Sample,
    scores: dict[str, float],
    timeline: Timeline,
) -> dict[str, object]:
    """Collect what the page shows for one sample."""
    telemetry = prediction.telemetry
    audio_seconds = sample.metadata.get("audio_seconds")
    tokens = [m.prompt_tokens for m in telemetry.models if m.prompt_tokens is not None]
    return {
        "id": prediction.sample_id,
        "status": prediction.status,
        "error": prediction.error,
        "audio_minutes": audio_seconds / 60 if audio_seconds else None,
        "prompt_tokens": sum(tokens) if tokens else None,
        "wall_seconds": telemetry.wall_seconds,
        "peak_rss_bytes": telemetry.peak_rss_bytes,
        "phases": [
            {"name": phase.name, "seconds": phase.seconds, "parent": phase.parent}
            for phase in telemetry.phases
        ],
        "processes": [
            {"name": entry.name, "peak": entry.peak_rss_bytes}
            for entry in telemetry.processes
        ],
        "memory": [_downsample(entry) for entry in timeline.memory],
        "windows": timeline.windows,
        "scores": scores,
        "note": _text(prediction.artifacts, "note"),
        "reference_note": _text(sample.artifacts, "note"),
        "link": f"{prediction.run_directory}/pipeline/",
    }


def _dataset_view(
    paths: RunPaths, plan: DatasetPlan, directories: list[Path], pipeline: Pipeline
) -> list[dict[str, object]]:
    """Collect every sample of one dataset."""
    name = plan.entry.name
    scores: dict[str, dict[str, float]] = {}
    for row in read_jsonl(paths.scores(name)):
        scores.setdefault(row["sample_id"], {})[row["metric"]] = row["value"]
    views = []
    for row in latest_rows(read_jsonl(paths.predictions(name))):
        prediction = load_prediction(paths.root, row)
        sample = plan.dataset.sample(prediction.sample_id)
        sample_scores = scores.get(prediction.sample_id, {})
        timeline = pipeline.timeline(directories, prediction.telemetry)
        views.append(_sample_view(prediction, sample, sample_scores, timeline))
    return views


def report_data(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    results: dict[str, object],
    pipeline: Pipeline,
) -> dict[str, object]:
    """Gather what the report displays."""
    manifest = read_json(paths.manifest)
    directories = execution_directories(paths.telemetry)
    datasets = {}
    for plan in plans:
        name = plan.entry.name
        datasets[name] = {
            **results["datasets"][name],
            "language": plan.dataset.language,
            "input": plan.entry.input,
            "samples": _dataset_view(paths, plan, directories, pipeline),
        }
    return {
        "name": manifest["name"],
        "created_at": manifest["created_at"],
        "pipeline": manifest["pipeline"].get("name", manifest["pipeline_type"]),
        "state": read_json(paths.status)["state"],
        "hardware": manifest["hardware"],
        "versions": manifest["versions"],
        "compute": results["compute"],
        "telemetry": results["telemetry"],
        "overall": results["overall"],
        "datasets": datasets,
    }


def _embed(data: dict[str, object]) -> str:
    """Serialize for an inline script, so no value can close the tag."""
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def write_report(
    paths: RunPaths,
    plans: tuple[DatasetPlan, ...],
    results: dict[str, object],
    pipeline: Pipeline,
) -> Path:
    """Write `report.html`, with everything inline so it opens anywhere.

    The page includes generated and reference notes, so treat it like the data.

    Returns:
        The report's path.
    """
    assets = files("scribe_eval") / "assets"
    page = (assets / "report.html").read_text(encoding="utf-8")
    page = page.replace("/*STYLE*/", (assets / "report.css").read_text("utf-8"))
    page = page.replace("/*SCRIPT*/", (assets / "report.js").read_text("utf-8"))
    page = page.replace(
        "__DATA__", _embed(report_data(paths, plans, results, pipeline))
    )
    paths.report.write_text(page, encoding="utf-8")
    return paths.report
