"""Small datasets and fake backends, so tests never load a model."""

import wave
from pathlib import Path

from scribnic import AudioFile, Note, Transcript
from scribnic.artifacts import StepOutput
from scribnic.config import ModelConfig, RuntimeConfig, StepConfig, WorkflowConfig

from scribe_eval.datasets.prepared import sample_directory, write_dataset, write_sample


def fake_workflow() -> WorkflowConfig:
    """Describe audio → transcript → note, with no real runtime."""
    runtime = RuntimeConfig("fake", "fake")
    speech = ModelConfig("speech", ("audio",), "transcript")
    note = ModelConfig("note", ("transcript",), "note")
    return WorkflowConfig(
        "fake",
        (
            StepConfig("speech", speech, runtime, "fake"),
            StepConfig("note", note, runtime, "fake"),
        ),
        inputs=("audio", "transcript"),
    )


class FakeBackend:
    """Transcribe to a fixed text and turn a transcript into a note."""

    def __init__(self, step: StepConfig, *, fail_notes_for: set[str], calls: list):
        """Keep which transcripts must fail, and record every call."""
        self.step = step
        self.fail_notes_for = fail_notes_for
        self.calls = calls

    def start(self) -> float:
        """Pretend the model is already loaded."""
        return 0.0

    def run(self, source) -> StepOutput:
        """Produce the next artifact."""
        self.calls.append(self.step.name)
        if isinstance(source, AudioFile):
            return StepOutput(Transcript.from_text(f"hello {source.path.parent.name}"))
        if source.text.split()[-1] in self.fail_notes_for:
            raise ValueError("Note rejected")
        text = "S — a\nO — b\nA — c\nP — d"
        usage = {"prompt_tokens": 10, "completion_tokens": 5}
        return StepOutput(Note(text), {"usage": usage})

    def process_ids(self) -> tuple[int, ...]:
        """Own no processes."""
        return ()

    def close(self) -> None:
        """Release nothing."""


def write_tiny_dataset(root: Path, sample_ids: list[str]) -> None:
    """Write samples whose reference transcript matches the fake backend."""
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
workflow = "unused"

[[datasets]]
name = "tiny"
split = "all"
input = "audio"
metrics = ["transcription", "note_format", "note_reference", "resources"]
"""
