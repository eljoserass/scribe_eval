"""Summarize run directories for a quick look at what is running or failed.

Reads only what runs saved: `manifest.json`, `status.json`, and prediction rows. A
run whose process died without updating its status still reads `running`; when the
status names a process on this machine that no longer exists, it shows as `stale`.
"""

import socket
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import psutil

from .storage import RunPaths, latest_rows, read_json, read_jsonl

ACTIVE_STATES = frozenset({"running", "scoring"})
_REASON_CHARACTERS = 160


class RunFilter(str, Enum):
    """Which runs to list."""

    RUNNING = "running"
    FAILED = "failed"
    ALL = "all"


@dataclass(frozen=True)
class RunSummary:
    """One run's progress, as its records stand. Holds no resources.

    Attributes:
        directory: The run directory.
        name: The experiment name.
        state: As recorded, or `stale` when its process is gone.
        done: Samples with a prediction row, across datasets.
        selected: Samples the run will process, across datasets.
        failed: Samples whose latest attempt failed.
        started_at: When the run was created.
        ended_at: When it last changed state, unless it is still active.
        reason: Why it failed: the run's error, or its first failed sample's.
    """

    directory: Path
    name: str
    state: str
    done: int
    selected: int
    failed: int
    started_at: datetime | None
    ended_at: datetime | None
    reason: str | None

    @property
    def active(self) -> bool:
        """Tell whether the run is still working."""
        return self.state in ACTIVE_STATES

    @property
    def has_failures(self) -> bool:
        """Tell whether the run or any of its samples failed."""
        return self.state in {"failed", "stale"} or self.failed > 0


def summarize_runs(
    runs_directory: Path, *, which: RunFilter = RunFilter.ALL
) -> list[RunSummary]:
    """Summarize the runs in a directory, newest first.

    Args:
        runs_directory: Parent of run directories, such as `runs/`.
        which: `running` keeps active runs; `failed` keeps runs that failed, went
            stale, or have failed samples; `all` keeps everything.

    Returns:
        The matching runs. Directories without a manifest and status are skipped.
    """
    if not runs_directory.is_dir():
        return []
    summaries = [
        summary
        for directory in runs_directory.iterdir()
        if (summary := summarize_run(directory)) is not None
    ]
    summaries.sort(key=_start_order, reverse=True)
    return [summary for summary in summaries if _matches(summary, which)]


def summarize_run(directory: Path) -> RunSummary | None:
    """Summarize one run, or return None if it isn't a run directory."""
    paths = RunPaths(directory)
    if not paths.manifest.is_file() or not paths.status.is_file():
        return None
    manifest = read_json(paths.manifest)
    status = read_json(paths.status)
    datasets = status.get("datasets", {})
    state = _effective_state(status)
    ended_at = None if state in ACTIVE_STATES else _timestamp(status.get("updated_at"))
    return RunSummary(
        directory=directory,
        name=manifest.get("name", directory.name),
        state=state,
        done=sum(dataset["run"] for dataset in datasets.values()),
        selected=sum(dataset["selected"] for dataset in datasets.values()),
        failed=sum(dataset["failed"] for dataset in datasets.values()),
        started_at=_timestamp(manifest.get("created_at")),
        ended_at=ended_at,
        reason=_reason(paths, status, datasets),
    )


def _effective_state(status: dict) -> str:
    """Mark an active run stale when its process on this machine is gone.

    Older runs record no process, so their state is taken as written.
    """
    state = status.get("state", "unknown")
    pid = status.get("pid")
    if state not in ACTIVE_STATES or pid is None:
        return state
    if status.get("host") != socket.gethostname():
        return state
    return state if psutil.pid_exists(pid) else "stale"


def _reason(paths: RunPaths, status: dict, datasets: dict) -> str | None:
    """Explain a failure briefly: the run's error, else the first failed sample's."""
    if status.get("error"):
        return _shorten(status["error"])
    for name, dataset in datasets.items():
        if not dataset["failed"]:
            continue
        for row in latest_rows(read_jsonl(paths.predictions(name))):
            if row["status"] != "success":
                message = row.get("error") or row["status"]
                return _shorten(f"{name}/{row['sample_id']}: {message}")
    return None


def _shorten(text: str) -> str:
    """Keep the first line of a message, cut to fit a table cell."""
    line = text.strip().splitlines()[0] if text.strip() else text
    if len(line) <= _REASON_CHARACTERS:
        return line
    return line[: _REASON_CHARACTERS - 1] + "…"


def _timestamp(value: str | None) -> datetime | None:
    """Parse an ISO timestamp, or return None."""
    return datetime.fromisoformat(value) if value else None


def _start_order(summary: RunSummary) -> float:
    """Sort key: start time, with unknown starts last."""
    return summary.started_at.timestamp() if summary.started_at else 0.0


def _matches(summary: RunSummary, which: RunFilter) -> bool:
    """Apply a filter."""
    if which is RunFilter.RUNNING:
        return summary.active
    if which is RunFilter.FAILED:
        return summary.has_failures
    return True


def states(summaries: Iterable[RunSummary]) -> dict[str, int]:
    """Count runs by state, for a one-line summary."""
    counts: dict[str, int] = {}
    for summary in summaries:
        counts[summary.state] = counts.get(summary.state, 0) + 1
    return counts
