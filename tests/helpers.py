"""A small dataset and a fake pipeline, so core tests need no real scribe."""

import wave
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from scribe_eval import PipelineResult, Timeline
from scribe_eval.artifacts import ArtifactKind, Note, Transcript
from scribe_eval.datasets.prepared import sample_directory, write_dataset, write_sample
from scribe_eval.telemetry import ModelUsage, Phase, SampleTelemetry


class FakePipeline:
    """The whole interface in a few lines, as a third party would write it."""

    def __init__(
        self,
        *,
        base_directory: Path = Path("."),
        fail_notes_for: Iterable[str] = (),
        interrupt_after: int | None = None,
        version: int = 1,
    ) -> None:
        """Choose which samples fail and when to simulate Ctrl+C."""
        self.fail_notes_for = set(fail_notes_for)
        self.interrupt_after = interrupt_after
        self.version = version
        self.runs = 0
        self.opened = False

    def describe(self) -> dict[str, object]:
        """Name this pipeline."""
        return {"name": "fake", "version": self.version}

    def produces(self, input_kind: ArtifactKind) -> set[ArtifactKind]:
        """Take audio or text."""
        if input_kind == "audio":
            return {"transcript", "note"}
        if input_kind == "transcript":
            return {"note"}
        raise ValueError(f"The fake pipeline does not accept {input_kind!r}.")

    @contextmanager
    def open(self, telemetry_directory: Path) -> Iterator[None]:
        """Pretend to hold resources."""
        self.opened = True
        yield
        self.opened = False

    def prepare(self, input_kind: ArtifactKind) -> dict[str, object]:
        """Report an instant load."""
        return {"elapsed_seconds": 0.0}

    def run(self, source, output_directory: Path) -> PipelineResult:
        """Transcribe to the sample's name, then write a note."""
        assert self.opened
        if self.interrupt_after is not None and self.runs >= self.interrupt_after:
            raise KeyboardInterrupt
        self.runs += 1
        name = source.path.parent.name
        transcript = Transcript.from_text(f"hello {name}")
        if name in self.fail_notes_for:
            return PipelineResult(
                "error", {"transcript": transcript}, error="Note rejected"
            )
        telemetry = SampleTelemetry(
            phases=(Phase("transcription", 1.0), Phase("note", 0.5)),
            models=(ModelUsage("note", 10, 5),),
        )
        artifacts = {
            "transcript": transcript,
            "note": Note("S — a\nO — b\nA — c\nP — d"),
        }
        return PipelineResult("success", artifacts, telemetry=telemetry)

    def timeline(
        self, telemetry_directories: Iterable[Path], telemetry: SampleTelemetry
    ) -> Timeline:
        """Have no timeline."""
        return Timeline()


def write_tiny_dataset(root: Path, sample_ids: list[str]) -> None:
    """Write samples whose reference transcript matches the fake pipeline."""
    for sample_id in sample_ids:
        directory = sample_directory(root, sample_id)
        directory.mkdir(parents=True)
        with wave.open(str(directory / "audio.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 16000)
        write_sample(
            root,
            sample_id,
            [Transcript.from_text(f"hello {sample_id}"), Note("S — a\nP — d")],
            {"audio_seconds": 1.0},
        )
    write_dataset(
        root, name=root.name, version="1", language="en", license="test", source="test"
    )


EXPERIMENT = """
name = "tiny"

[pipeline]
type = "tests.helpers:FakePipeline"

[[datasets]]
name = "tiny"
split = "all"
input = "audio"
metrics = ["transcription", "note_format", "note_reference", "resources"]
"""
