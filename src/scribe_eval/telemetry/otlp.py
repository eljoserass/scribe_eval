"""Read OpenTelemetry JSONL files. Nothing here knows who exported them."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import fmean

from ..storage import read_jsonl


@dataclass(frozen=True)
class Span:
    """A finished span, with its attributes as plain values."""

    name: str
    trace_id: str
    start_ns: int
    end_ns: int
    attributes: dict[str, object]

    @property
    def seconds(self) -> float:
        """Span duration."""
        return (self.end_ns - self.start_ns) / 1e9


@dataclass(frozen=True)
class Point:
    """One metric data point."""

    time_ns: int
    value: float
    attributes: dict[str, object]


def _attributes(entries: Iterable[dict] | None) -> dict[str, object]:
    """Unwrap OTLP's typed attribute values."""
    return {
        entry["key"]: next(iter(entry["value"].values())) for entry in entries or []
    }


def _version(path: Path) -> tuple[int, int]:
    """Identify a file's contents cheaply, so a cached read can't go stale."""
    if not path.exists():
        return 0, 0
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=4)
def _file_spans(path: Path, version: tuple[int, int]) -> tuple[Span, ...]:
    """Parse one traces file. A long run reads it once per sample otherwise."""
    return tuple(
        Span(
            span["name"],
            span["traceId"],
            int(span["startTimeUnixNano"]),
            int(span["endTimeUnixNano"]),
            _attributes(span.get("attributes")),
        )
        for row in read_jsonl(path)
        for resource in row.get("resourceSpans", [])
        for scope in resource.get("scopeSpans", [])
        for span in scope.get("spans", [])
    )


def read_spans(directories: Iterable[Path]) -> list[Span]:
    """Read every span from `traces.jsonl` in each directory."""
    spans = []
    for directory in directories:
        path = directory / "traces.jsonl"
        spans += _file_spans(path, _version(path))
    return spans


@lru_cache(maxsize=4)
def _file_points(
    path: Path, version: tuple[int, int], metric_name: str
) -> tuple[Point, ...]:
    """Parse one metric from a metrics file, which can be hundreds of megabytes."""
    points = []
    for row in read_jsonl(path):
        for resource in row.get("resourceMetrics", []):
            for scope in resource.get("scopeMetrics", []):
                for metric in scope.get("metrics", []):
                    if metric["name"] == metric_name:
                        points += _metric_points(metric)
    return tuple(points)


def read_points(directories: Iterable[Path], metric_name: str) -> list[Point]:
    """Read the gauge or sum points of one metric from `metrics.jsonl`."""
    points = []
    for directory in directories:
        path = directory / "metrics.jsonl"
        points += _file_points(path, _version(path), metric_name)
    return points


def _metric_points(metric: dict) -> list[Point]:
    """Unwrap a metric's data points; histograms aren't used here."""
    data = metric.get("gauge") or metric.get("sum") or {}
    return [
        Point(
            int(point["timeUnixNano"]),
            float(point.get("asInt", point.get("asDouble", 0))),
            _attributes(point.get("attributes")),
        )
        for point in data.get("dataPoints", [])
    ]


def execution_directories(root: Path) -> list[Path]:
    """List a run's telemetry folders, one per execution."""
    return sorted(path for path in root.glob("execution-*") if path.is_dir())


def summarize(root: Path) -> dict[str, object]:
    """Reduce a run's telemetry to time per span and peak machine memory.

    `system.memory.usage` covers everything on the machine, including what was
    already running, so it's context rather than the cost of the pipeline.

    Args:
        root: The run's `telemetry` directory.

    Returns:
        A summary, empty if nothing was exported.
    """
    directories = execution_directories(root)
    durations: dict[str, list[float]] = defaultdict(list)
    for span in read_spans(directories):
        durations[span.name].append(span.seconds)
    memory = [
        point.value
        for point in read_points(directories, "system.memory.usage")
        if point.attributes.get("state") == "used"
    ]
    return {
        "spans": {
            name: {
                "count": len(values),
                "total_seconds": sum(values),
                "mean_seconds": fmean(values),
            }
            for name, values in sorted(durations.items())
        },
        "machine_memory_bytes": {
            "baseline": min(memory, default=None),
            "peak": max(memory, default=None),
        },
    }
