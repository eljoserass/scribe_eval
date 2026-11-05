"""Translate artifacts between scribe_eval and Scribnic."""

import scribnic
from scribnic.artifacts import SpeakerSegment, TextSegment

from ...artifacts import (
    Artifact,
    ArtifactKind,
    AudioFile,
    DiarizedTranscript,
    Note,
    Segment,
    Transcript,
)


def to_scribnic(artifact: Artifact) -> scribnic.Artifact:
    """Convert an input for Scribnic."""
    if isinstance(artifact, AudioFile):
        return scribnic.AudioFile(artifact.path)
    if isinstance(artifact, Transcript):
        return scribnic.Transcript(
            tuple(TextSegment(s.text, s.start, s.end) for s in artifact.segments)
        )
    if isinstance(artifact, DiarizedTranscript):
        return scribnic.DiarizedTranscript(
            tuple(
                SpeakerSegment(s.text, s.speaker, s.start, s.end, s.role)
                for s in artifact.segments
            )
        )
    return scribnic.Note(artifact.text)


def from_scribnic(artifact: scribnic.Artifact) -> Artifact:
    """Convert an output from Scribnic."""
    if isinstance(artifact, scribnic.Note):
        return Note(artifact.text)
    if isinstance(artifact, scribnic.AudioFile):
        return AudioFile(artifact.path)
    if isinstance(artifact, scribnic.DiarizedTranscript):
        return DiarizedTranscript(
            tuple(
                Segment(s.text, s.speaker, s.start, s.end, s.role)
                for s in artifact.segments
            )
        )
    return Transcript(
        tuple(Segment(s.text, start=s.start, end=s.end) for s in artifact.segments)
    )


def produced_artifacts(result: scribnic.RunResult) -> dict[ArtifactKind, Artifact]:
    """Keep what a run produced. `RunResult.artifacts` also carries the input."""
    return {
        artifact.kind: from_scribnic(artifact)
        for artifact in result.artifacts
        if artifact.kind != result.source.kind
    }
