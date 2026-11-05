"""Typer commands: parse arguments, call the application layer, show results."""

import os
from pathlib import Path
from typing import Annotated

import typer
from scribnic import WorkflowRunError

from .application import prepare_dataset, resume_run, run_experiment, score_run
from .terminal import TerminalView

app = typer.Typer(
    help="Evaluate Scribnic workflows on clinical datasets.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

DataOption = Annotated[
    Path | None,
    typer.Option(help="Prepared datasets. Defaults to $SCRIBE_EVAL_DATA, then data/"),
]
OfflineOption = Annotated[bool, typer.Option("--offline", help="Require cached models")]


def _data_directory(option: Path | None) -> Path:
    """Pick the data root."""
    return option or Path(os.getenv("SCRIBE_EVAL_DATA", "data"))


def _download(offline: bool) -> bool | None:
    """Translate the offline flag into Scribnic's download setting."""
    return False if offline else None


def _fail(error: Exception) -> typer.Exit:
    """Show an expected error and return the exit to raise."""
    TerminalView().progress.print(f"Error: {error}", markup=False)
    return typer.Exit(1)


@app.command()
def prepare(
    name: Annotated[str, typer.Argument(help="Dataset: primock57 or smoke")],
    *,
    source: Annotated[Path | None, typer.Option(help="Local raw data")] = None,
    sample_id: Annotated[
        list[str] | None, typer.Option("--id", help="Convert only this sample")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Convert the first N")] = None,
    data_dir: DataOption = None,
) -> None:
    """Convert a dataset to the common format, downloading it if needed."""
    ids = tuple(sample_id) if sample_id else None
    try:
        dataset = prepare_dataset(
            name, _data_directory(data_dir), source=source, ids=ids, limit=limit
        )
    except (ValueError, OSError) as error:
        raise _fail(error) from error
    print(f"{dataset.name}: {len(dataset.splits['all'])} samples in {dataset.root}")


@app.command()
def run(
    experiment: Annotated[Path, typer.Argument(help="Experiment TOML")],
    *,
    output_dir: Annotated[Path | None, typer.Option(help="Exact run directory")] = None,
    runs_dir: Annotated[Path, typer.Option(help="Parent of new runs")] = Path("runs"),
    data_dir: DataOption = None,
    offline: OfflineOption = False,
) -> None:
    """Run an experiment and score it."""
    view = TerminalView()
    try:
        directory, results = run_experiment(
            experiment,
            data_directory=_data_directory(data_dir),
            runs_directory=runs_dir,
            run_directory=output_dir,
            download=_download(offline),
            on_prediction=view.on_prediction,
        )
    except (ValueError, OSError, RuntimeError, WorkflowRunError) as error:
        raise _fail(error) from error
    except KeyboardInterrupt:
        view.progress.print("Interrupted. Continue with `scribe-eval resume`.")
        raise typer.Exit(130) from None
    view.results(results)
    print(f"Run: {directory}")


@app.command()
def resume(
    run_directory: Annotated[Path, typer.Argument(help="Existing run")],
    *,
    retry_errors: Annotated[
        bool, typer.Option("--retry-errors", help="Run failed samples again")
    ] = False,
    data_dir: DataOption = None,
    offline: OfflineOption = False,
) -> None:
    """Finish a run with the same samples and configuration."""
    view = TerminalView()
    try:
        results = resume_run(
            run_directory,
            data_directory=_data_directory(data_dir),
            retry_errors=retry_errors,
            download=_download(offline),
            on_prediction=view.on_prediction,
        )
    except (ValueError, OSError, RuntimeError, WorkflowRunError) as error:
        raise _fail(error) from error
    except KeyboardInterrupt:
        view.progress.print("Interrupted. Continue with `scribe-eval resume`.")
        raise typer.Exit(130) from None
    view.results(results)


@app.command()
def score(
    run_directory: Annotated[Path, typer.Argument(help="Existing run")],
    *,
    data_dir: DataOption = None,
) -> None:
    """Recompute metrics from saved predictions, without running any model."""
    try:
        results = score_run(run_directory, data_directory=_data_directory(data_dir))
    except (ValueError, OSError) as error:
        raise _fail(error) from error
    TerminalView().results(results)


def main() -> None:
    """Run the CLI."""
    app(prog_name="scribe-eval")


if __name__ == "__main__":
    main()
