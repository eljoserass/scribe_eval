"""Run a Scribnic workflow behind the `Pipeline` interface."""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from scribnic import WorkflowConfig, WorkflowRunError, load_workflow, open_workflow
from scribnic.planning import resolve_plan, validate_workflow
from scribnic.telemetry import Telemetry
from scribnic.workflow import BackendFactory, WorkflowSession

from ...artifacts import Artifact, ArtifactKind
from ...telemetry import SampleTelemetry
from ..base import PipelineResult, Timeline
from .conversion import produced_artifacts, to_scribnic
from .telemetry import memory_series, phase_windows, sample_telemetry, started_at


class ScribnicPipeline:
    """One Scribnic workflow. Holds a session, and its models, inside `open`."""

    def __init__(
        self,
        *,
        base_directory: Path,
        workflow: str | WorkflowConfig,
        download: bool | None = None,
        backend_factory: BackendFactory | None = None,
    ) -> None:
        """Read the workflow without loading models.

        Args:
            base_directory: Resolves a relative `workflow` file.
            workflow: Preset name, TOML file, or parsed configuration.
            download: False requires cached models.
            backend_factory: Replaces model backends, for tests.
        """
        if isinstance(workflow, str) and (base_directory / workflow).exists():
            workflow = str((base_directory / workflow).resolve())
        self.config = load_workflow(workflow)
        validate_workflow(self.config)
        self.download = download
        self.backend_factory = backend_factory
        self._session: WorkflowSession | None = None

    def describe(self) -> dict[str, object]:
        """Return the effective workflow."""
        return self.config.to_dict()

    def produces(self, input_kind: ArtifactKind) -> set[ArtifactKind]:
        """Ask Scribnic's planner what this input would run."""
        plan = resolve_plan(self.config, input_kind, self.config.default_output)
        return {planned.step.model.output for planned in plan}

    @contextmanager
    def open(self, telemetry_directory: Path) -> Iterator[None]:
        """Export telemetry and keep the session open."""
        with ExitStack() as stack:
            stack.enter_context(Telemetry(telemetry_directory))
            self._session = stack.enter_context(
                open_workflow(
                    self.config,
                    backend_factory=self.backend_factory,
                    download=self.download,
                )
            )
            try:
                yield
            finally:
                self._session = None

    def prepare(self, input_kind: ArtifactKind) -> dict[str, object]:
        """Load the steps this input needs."""
        return self._session.start(source_kind=input_kind)

    def run(self, source: Artifact, output_directory: Path) -> PipelineResult:
        """Run one input; Scribnic keeps its own full record in the directory."""
        try:
            result = self._session.run(to_scribnic(source), output_dir=output_directory)
        except WorkflowRunError as error:
            result = error.result
        return PipelineResult(
            status="success" if result.status == "success" else "error",
            artifacts=produced_artifacts(result),
            error=result.error,
            telemetry=sample_telemetry(result),
        )

    def timeline(
        self, telemetry_directories: Iterable[Path], telemetry: SampleTelemetry
    ) -> Timeline:
        """Rebuild memory and step timing from the OpenTelemetry export."""
        directories = list(telemetry_directories)
        return Timeline(
            memory_series(directories, telemetry),
            phase_windows(directories, telemetry),
            started_at(directories, telemetry),
        )
