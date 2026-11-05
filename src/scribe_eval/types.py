"""Records shared by datasets, inference, metrics, and storage."""

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from .artifacts import Artifact, ArtifactKind, DiarizedTranscript
from .telemetry import SampleTelemetry


def find_artifact(
    artifacts: Mapping[ArtifactKind, Artifact], kind: ArtifactKind
) -> Artifact | None:
    """Return an artifact, deriving a plain transcript from a diarized one."""
    if kind in artifacts:
        return artifacts[kind]
    diarized = artifacts.get("diarized_transcript")
    if kind == "transcript" and isinstance(diarized, DiarizedTranscript):
        return diarized.without_speakers()
    return None


def satisfies(available: Container[ArtifactKind], kind: ArtifactKind) -> bool:
    """Tell whether a set of artifact kinds can provide the requested kind."""
    if kind in available:
        return True
    return kind == "transcript" and "diarized_transcript" in available


@dataclass(frozen=True)
class Sample:
    """One consultation and the reference artifacts its dataset provides."""

    id: str
    split: str
    language: str
    artifacts: dict[ArtifactKind, Artifact]
    metadata: dict[str, object] = field(default_factory=dict)

    def artifact(self, kind: ArtifactKind) -> Artifact:
        """Return a reference artifact.

        Raises:
            KeyError: If the sample cannot provide that kind.
        """
        artifact = find_artifact(self.artifacts, kind)
        if artifact is None:
            raise KeyError(f"Sample {self.id!r} has no {kind!r} artifact.")
        return artifact


@dataclass(frozen=True)
class Prediction:
    """What one workflow execution produced for a sample, even when it failed."""

    dataset: str
    sample_id: str
    variant: str
    repeat: int
    input_kind: ArtifactKind
    status: Literal["success", "error"]
    error: str | None
    artifacts: dict[ArtifactKind, Artifact]
    telemetry: SampleTelemetry
    run_directory: str
    telemetry_directory: str

    @property
    def key(self) -> tuple[str, str, int]:
        """Identify this execution within its dataset."""
        return self.variant, self.sample_id, self.repeat


@dataclass(frozen=True)
class Requires:
    """Artifact kinds a metric needs from the workflow and from the dataset."""

    predicted: tuple[ArtifactKind, ...] = ()
    reference: tuple[ArtifactKind, ...] = ()


@dataclass(frozen=True)
class Score:
    """One metric's value for one prediction."""

    metric: str
    value: float
    details: dict[str, float] = field(default_factory=dict)


class Metric(Protocol):
    """Score a prediction and define how those scores combine."""

    name: str
    group: str
    version: int
    requires: Requires

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one prediction, or return None when it does not apply."""
        ...

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Combine the scores of one dataset."""
        ...
