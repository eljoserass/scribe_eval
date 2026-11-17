"""Terminal presentation. The core never imports this."""

from datetime import datetime

from rich.console import Console
from rich.progress_bar import ProgressBar
from rich.table import Table

from .comparison import Comparison, DatasetComparison
from .listing import RunSummary, states
from .types import Prediction


class TerminalView:
    """Print progress and results."""

    def __init__(self) -> None:
        """Write to stderr for progress, keeping stdout for results."""
        self.progress = Console(stderr=True)
        self.output = Console()

    def on_prediction(self, prediction: Prediction, position: int, total: int) -> None:
        """Print one finished sample."""
        label = f"[{position}/{total}] {prediction.dataset}/{prediction.sample_id}"
        seconds = f"{prediction.telemetry.wall_seconds:.1f} s"
        if prediction.status == "success":
            self.progress.print(f"{label} ok, {seconds}", markup=False)
        else:
            self.progress.print(f"{label} failed: {prediction.error}", markup=False)

    def results(self, results: dict[str, object]) -> None:
        """Print one table per dataset."""
        for name, dataset in results["datasets"].items():
            run, selected = dataset["n_run"], dataset["n_selected"]
            title = f"{name}: {run}/{selected} run, {dataset['n_failed']} failed"
            table = Table(title=title, title_justify="left")
            table.add_column("Metric")
            table.add_column("n", justify="right")
            table.add_column("Values")
            for metric, values in dataset["metrics"].items():
                shown = ", ".join(
                    f"{key} {value:.4g}" for key, value in values.items() if key != "n"
                )
                table.add_row(metric, str(values["n"]), shown)
            for metric, reason in dataset["not_applicable"].items():
                table.add_row(metric, "n/a", reason)
            self.output.print(table)
            for sample_id, error in dataset["failures"].items():
                self.output.print(f"Failed {sample_id}: {error}", markup=False)

    def runs(self, summaries: list[RunSummary], *, show_reasons: bool) -> None:
        """Print runs as a table: name, state, progress, and times.

        Args:
            summaries: Runs to show, already filtered and sorted.
            show_reasons: Add a column saying why each run failed.
        """
        table = Table(caption=_runs_caption(summaries), caption_justify="left")
        table.add_column("Run", no_wrap=True)
        table.add_column("State", no_wrap=True)
        table.add_column("Progress", no_wrap=True)
        table.add_column("Failed", justify="right")
        table.add_column("Started", no_wrap=True)
        table.add_column("Ended", no_wrap=True)
        if show_reasons:
            table.add_column("Reason", overflow="fold", min_width=20)
        for summary in summaries:
            row = [
                _run_label(summary.directory.name),
                _state_text(summary.state),
                _progress(summary),
                str(summary.failed),
                _time_text(summary.started_at),
                _time_text(summary.ended_at) if not summary.active else "running",
            ]
            if show_reasons:
                row.append(summary.reason or "")
            table.add_row(*row)
        self.output.print(table)

    def no_runs(self, every_run: list[RunSummary], *, which: str) -> None:
        """Say that nothing matched, with a count of runs by state."""
        counts = ", ".join(
            f"{count} {state}" for state, count in states(every_run).items()
        )
        suffix = f" Runs here: {counts}." if counts else ""
        self.output.print(f"No {which} runs.{suffix}", markup=False)

    def comparison(self, comparison: Comparison) -> None:
        """Print a legend of runs, then one table per dataset with a column per run.

        Coverage comes first in each table, so a run that failed more never reads as
        better on the remaining samples.
        """
        labels = [run.label for run in comparison.runs]
        for index, run in enumerate(comparison.runs, start=1):
            pipeline = run.manifest["pipeline"].get(
                "name", run.manifest["pipeline_type"]
            )
            self.output.print(f"[{index}] {run.label}: {pipeline}", markup=False)
        for name, dataset in comparison.datasets.items():
            self.output.print(_comparison_table(name, dataset, labels))
        for warning in comparison.warnings:
            self.output.print(f"Warning: {warning}", markup=False)


def _comparison_table(
    name: str, dataset: DatasetComparison, labels: list[str]
) -> Table:
    """Build one dataset's table; a dash marks a value a run doesn't have."""
    table = Table(title=name, title_justify="left")
    table.add_column("Metric")
    table.add_column("Statistic")
    for index in range(1, len(labels) + 1):
        table.add_column(f"[{index}]", justify="right")

    coverage = []
    for label in labels:
        counts = dataset.coverage.get(label)
        if counts is None:
            coverage.append("-")
        else:
            run, selected, failed = (
                counts["n_run"],
                counts["n_selected"],
                counts["n_failed"],
            )
            coverage.append(f"{run}/{selected}, {failed} failed")
    table.add_row("samples", "run, failed", *coverage, end_section=True)

    for (metric, statistic), values in dataset.values.items():
        shown = [_format_value(values.get(label)) for label in labels]
        table.add_row(metric, statistic, *shown)
    for metric, reasons in dataset.not_applicable.items():
        shown = ["n/a" if label in reasons else "-" for label in labels]
        table.add_row(metric, "", *shown)
    return table


def _format_value(value: float | None) -> str:
    """Show four significant digits, or a dash when the run has no value."""
    if value is None:
        return "-"
    return f"{value:.4g}"


_STATE_STYLES = {
    "running": "cyan",
    "scoring": "cyan",
    "completed": "green",
    "predicted": "green",
    "failed": "red",
    "stale": "red",
    "interrupted": "yellow",
}


def _state_text(state: str) -> str:
    """Color a state for the table."""
    style = _STATE_STYLES.get(state)
    return f"[{style}]{state}[/{style}]" if style else state


def _progress(summary: RunSummary) -> Table:
    """Draw a progress bar with its count beside it."""
    grid = Table.grid(padding=(0, 1))
    total = max(summary.selected, 1)
    grid.add_row(
        ProgressBar(total=total, completed=summary.done, width=10),
        f"{summary.done}/{summary.selected}",
    )
    return grid


def _time_text(moment: datetime | None) -> str:
    """Show a timestamp in local time, to the minute."""
    if moment is None:
        return "—"
    return moment.astimezone().strftime("%m-%d %H:%M")


def _run_label(directory_name: str) -> str:
    """Drop the leading date and time from a run directory name; Started shows it."""
    parts = directory_name.split("-", 2)
    has_stamp = len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit()
    return parts[2] if has_stamp else directory_name


def _runs_caption(summaries: list[RunSummary]) -> str:
    """Say where the listed runs live, since names omit their timestamp."""
    parents = {str(summary.directory.parent) for summary in summaries}
    return f"In {parents.pop()}/" if len(parents) == 1 else ""
