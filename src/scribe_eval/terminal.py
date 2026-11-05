"""Terminal presentation. The core never imports this."""

from rich.console import Console
from rich.table import Table

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
        seconds = f"{prediction.elapsed_seconds:.1f} s"
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
