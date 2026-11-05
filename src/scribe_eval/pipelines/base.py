"""The interface a scribe implements to be evaluated."""

from collections.abc import Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from ..artifacts import Artifact, ArtifactKind
from ..telemetry import MemorySeries, SampleTelemetry


@dataclass(frozen=True)
class PipelineResult:
    """What a pipeline did with one input.

    `artifacts` holds what it produced, never its input. A failed run keeps whatever
    it produced before failing. Telemetry is optional beyond the default; the harness
    sets `wall_seconds` itself.
    """

    status: Literal["success", "error"]
    artifacts: dict[ArtifactKind, Artifact] = field(default_factory=dict)
    error: str | None = None
    telemetry: SampleTelemetry = SampleTelemetry()


@dataclass(frozen=True)
class Timeline:
    """How one sample unfolded, for charts. Empty if a pipeline can't say."""

    memory: tuple[MemorySeries, ...] = ()
    windows: tuple[tuple[str, float, float], ...] = ()  # name, start, end in seconds


class Pipeline(Protocol):
    """A scribe. Constructing one must be cheap; models load inside `open`."""

    def describe(self) -> dict[str, object]:
        """Return the effective configuration, with a `name` key.

        Saved in the manifest. A resumed run must describe itself identically.
        """
        ...

    def produces(self, input_kind: ArtifactKind) -> set[ArtifactKind]:
        """List what this pipeline produces from an input, excluding the input.

        Raises:
            ValueError: If it doesn't accept that input.
        """
        ...

    def open(self, telemetry_directory: Path) -> AbstractContextManager[None]:
        """Hold resources for a batch of samples.

        Args:
            telemetry_directory: Where to export telemetry, if the pipeline has any.
        """
        ...

    def prepare(self, input_kind: ArtifactKind) -> dict[str, object]:
        """Load what this input needs, returning load times and the like."""
        ...

    def run(self, source: Artifact, output_directory: Path) -> PipelineResult:
        """Process one input. A failure is a result, not an exception.

        Args:
            source: The input.
            output_directory: Unused path for anything the pipeline wants to keep.
        """
        ...

    def timeline(
        self, telemetry_directories: Iterable[Path], telemetry: SampleTelemetry
    ) -> Timeline:
        """Rebuild a sample's timeline from exported telemetry, for the report."""
        ...
