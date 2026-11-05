"""What flows through a scribe: audio, transcripts, and notes.

Pipelines translate between these and their own types. The JSON form is plain, so
datasets and predictions stay readable without any pipeline installed.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal, TypeAlias

ArtifactKind: TypeAlias = Literal["audio", "transcript", "diarized_transcript", "note"]
ARTIFACT_KINDS: tuple[ArtifactKind, ...] = (
    "audio",
    "transcript",
    "diarized_transcript",
    "note",
)


@dataclass(frozen=True)
class AudioFile:
    """A recording on disk, as mono PCM16 WAV at 16 kHz."""

    path: Path
    kind: ClassVar[ArtifactKind] = "audio"


@dataclass(frozen=True)
class Segment:
    """A piece of a transcript. Times are in seconds, and may be unknown."""

    text: str
    speaker: str | None = None
    start: float | None = None
    end: float | None = None
    role: str | None = None


@dataclass(frozen=True)
class Transcript:
    """Text with no reliable speaker attribution."""

    segments: tuple[Segment, ...]
    kind: ClassVar[ArtifactKind] = "transcript"

    @classmethod
    def from_text(cls, text: str) -> "Transcript":
        """Wrap plain text."""
        return cls((Segment(text),))

    @property
    def text(self) -> str:
        """The whole text."""
        return "\n".join(segment.text for segment in self.segments)


@dataclass(frozen=True)
class DiarizedTranscript:
    """Text attributed to speakers."""

    segments: tuple[Segment, ...]
    kind: ClassVar[ArtifactKind] = "diarized_transcript"

    @property
    def text(self) -> str:
        """The whole text."""
        return "\n".join(segment.text for segment in self.segments)

    def without_speakers(self) -> Transcript:
        """Drop identities, keeping text, turns, and times."""
        return Transcript(
            tuple(
                Segment(segment.text, start=segment.start, end=segment.end)
                for segment in self.segments
            )
        )


@dataclass(frozen=True)
class Note:
    """A clinical note."""

    text: str
    kind: ClassVar[ArtifactKind] = "note"


Artifact: TypeAlias = AudioFile | Transcript | DiarizedTranscript | Note


def artifact_kind(value: str) -> ArtifactKind:
    """Check an artifact kind from a config file.

    Raises:
        ValueError: If the kind doesn't exist.
    """
    if value not in ARTIFACT_KINDS:
        raise ValueError(f"Unknown artifact kind: {value!r}")
    return value


def save_artifact(artifact: Artifact, path: Path) -> None:
    """Write an artifact as JSON."""
    data = {"kind": artifact.kind, **asdict(artifact)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def load_artifact(path: Path) -> Artifact:
    """Read an artifact from JSON.

    Raises:
        ValueError: If the file holds an unknown kind.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    kind = data.get("kind")
    if kind == "note":
        return Note(data["text"])
    if kind == "audio":
        return AudioFile(Path(data["path"]))
    segments = tuple(
        Segment(
            entry["text"],
            entry.get("speaker"),
            entry.get("start"),
            entry.get("end"),
            entry.get("role"),
        )
        for entry in data.get("segments", [])
    )
    if kind == "transcript":
        return Transcript(segments)
    if kind == "diarized_transcript":
        return DiarizedTranscript(segments)
    raise ValueError(f"Unknown artifact in {path}: {kind!r}")
