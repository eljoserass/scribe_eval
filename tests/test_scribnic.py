"""Check the Scribnic implementation, the only tests that need Scribnic."""

import tempfile
import unittest
from pathlib import Path

import scribnic
from scribnic import RunResult
from scribnic.artifacts import StepOutput, StepRecord
from scribnic.config import ModelConfig, RuntimeConfig, StepConfig, WorkflowConfig

from scribe_eval import run_experiment
from scribe_eval.artifacts import DiarizedTranscript, Segment, Transcript
from scribe_eval.pipelines.scribnic import ScribnicPipeline
from scribe_eval.pipelines.scribnic.conversion import from_scribnic, to_scribnic
from scribe_eval.pipelines.scribnic.telemetry import (
    memory_series,
    runtime_device,
    sample_telemetry,
)
from scribe_eval.storage import append_jsonl, read_jsonl
from scribe_eval.telemetry import ProcessUsage, SampleTelemetry

from .helpers import EXPERIMENT, write_tiny_dataset


def workflow() -> WorkflowConfig:
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
    """Stand in for a model inside Scribnic."""

    def __init__(self, step: StepConfig):
        """Keep the step."""
        self.step = step

    def start(self) -> float:
        """Pretend the model is loaded."""
        return 0.0

    def run(self, source) -> StepOutput:
        """Produce the next artifact."""
        if isinstance(source, scribnic.AudioFile):
            name = source.path.parent.name
            return StepOutput(scribnic.Transcript.from_text(f"hello {name}"))
        if source.text.endswith("b"):
            raise ValueError("Note rejected")
        return StepOutput(scribnic.Note("S — a\nO — b\nA — c\nP — d"))

    def process_ids(self) -> tuple[int, ...]:
        """Own no processes."""
        return ()

    def close(self) -> None:
        """Release nothing."""


def result() -> RunResult:
    """Build a result shaped like a real Scribnic run."""
    note = {
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "timings": {
            "prompt_ms": 2000.0,
            "predicted_ms": 1000.0,
            "prompt_per_second": 50.0,
            "predicted_per_second": 20.0,
        },
        "runtime": {"argv": ["llama-server", "-ngl", "0"]},
    }
    speech = {"runtime": {"argv": ["whisper-server", "--no-gpu"]}}
    processes = [
        {"name": "llama-server", "peak_sampled_rss_bytes": 9, "average_cpu_cores": 1.5}
    ]
    return RunResult(
        run_id="run",
        workflow="w",
        requested_output="note",
        source=scribnic.AudioFile("a.wav"),
        artifacts=(),
        steps=(
            StepRecord("asr_step", "b", 5.0, speech),
            StepRecord("write", "b", 3.5, note),
        ),
        configuration={
            "steps": [
                {"name": "asr_step", "model": {"output": "diarized_transcript"}},
                {"name": "write", "model": {"output": "note"}},
            ]
        },
        started_at="now",
        elapsed_seconds=9.0,
        trace_id="trace",
        resources={"peak_sampled_process_rss_sum_bytes": 12, "processes": processes},
    )


class TelemetryTests(unittest.TestCase):
    def test_step_names_become_shared_phase_names(self):
        phases = sample_telemetry(result()).phases
        self.assertEqual(
            [(phase.name, phase.parent) for phase in phases],
            [
                ("transcription", None),
                ("note", None),
                ("prefill", "note"),
                ("generation", "note"),
            ],
        )

    def test_round_trips_through_a_row(self):
        telemetry = sample_telemetry(result())
        self.assertEqual(SampleTelemetry.from_dict(telemetry.to_dict()), telemetry)

    def test_devices_come_from_the_actual_command(self):
        self.assertEqual(runtime_device({"argv": ["x", "--no-gpu"]}), "cpu")
        self.assertEqual(runtime_device({"argv": ["x", "-ngl", "0"]}), "cpu")
        self.assertEqual(runtime_device({"argv": ["x", "-ngl", "all"]}), "accelerator")
        self.assertEqual(
            runtime_device({"argv": ["x", "--device", "cuda"]}), "accelerator"
        )
        self.assertIsNone(runtime_device({"argv": ["x"]}))
        self.assertIsNone(runtime_device(None))


class MemorySeriesTests(unittest.TestCase):
    def test_points_after_the_sample_are_dropped(self):
        def point(seconds: int, value: str) -> dict:
            """Build one gauge point."""
            attributes = [
                {"key": "scribnic.run.id", "value": {"stringValue": "run"}},
                {"key": "process.pid", "value": {"intValue": "7"}},
            ]
            return {
                "attributes": attributes,
                "timeUnixNano": str(seconds * 10**9),
                "asInt": value,
            }

        points = [point(0, "1"), point(5, "2"), point(50, "2")]
        metric = {
            "name": "scribnic.process.memory.usage",
            "gauge": {"dataPoints": points},
        }
        row = {"resourceMetrics": [{"scopeMetrics": [{"metrics": [metric]}]}]}
        telemetry = SampleTelemetry(
            wall_seconds=10.0,
            processes=(ProcessUsage("llm", 2, 1.0, 7),),
            run_id="run",
        )
        with tempfile.TemporaryDirectory() as directory:
            append_jsonl(Path(directory) / "metrics.jsonl", row)
            series = memory_series([Path(directory)], telemetry)
        self.assertEqual(series[0].process, "llm")
        self.assertEqual(series[0].bytes, (1, 2))


class ConversionTests(unittest.TestCase):
    def test_transcripts_survive_the_round_trip(self):
        segment = Segment("hola", "S1", 0.0, 1.0, "doctor")
        diarized = DiarizedTranscript((segment,))
        self.assertEqual(from_scribnic(to_scribnic(diarized)), diarized)
        plain = Transcript((Segment("hola", start=0.0, end=1.0),))
        self.assertEqual(from_scribnic(to_scribnic(plain)), plain)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        write_tiny_dataset(self.directory / "data" / "tiny", ["a", "b"])
        self.experiment = self.directory / "experiment.toml"
        self.experiment.write_text(EXPERIMENT, encoding="utf-8")
        self.run_directory = self.directory / "run"

    def pipeline(self) -> ScribnicPipeline:
        """Build the real implementation on fake backends."""
        return ScribnicPipeline(
            base_directory=self.directory,
            workflow=workflow(),
            backend_factory=FakeBackend,
        )

    def run_it(self) -> dict:
        """Run the experiment through Scribnic."""
        _, results = run_experiment(
            self.experiment,
            data_directory=self.directory / "data",
            run_directory=self.run_directory,
            pipeline=self.pipeline(),
        )
        return results

    def rows(self) -> list[dict]:
        """Read predictions."""
        path = self.run_directory / "datasets" / "tiny" / "predictions.jsonl"
        return read_jsonl(path)

    def test_produces_excludes_the_input(self):
        self.assertEqual(self.pipeline().produces("transcript"), {"note"})
        self.assertEqual(self.pipeline().produces("audio"), {"transcript", "note"})

    def test_rejected_input_is_a_validation_error(self):
        with self.assertRaises(ValueError):
            self.pipeline().produces("note")

    def test_the_input_is_not_recorded_as_produced(self):
        self.run_it()
        self.assertEqual(self.rows()[0]["artifacts"], ["note", "transcript"])

    def test_failure_keeps_partial_artifacts(self):
        results = self.run_it()
        self.assertEqual(results["datasets"]["tiny"]["n_failed"], 1)
        self.assertEqual(self.rows()[1]["artifacts"], ["transcript"])
        self.assertEqual(results["datasets"]["tiny"]["metrics"]["wer"]["n"], 2)

    def test_telemetry_is_exported_and_linked_to_each_sample(self):
        results = self.run_it()
        traces = self.run_directory / "telemetry" / "execution-1" / "traces.jsonl"
        self.assertTrue(traces.exists())
        self.assertIsNotNone(self.rows()[0]["telemetry"]["trace_id"])
        spans = results["telemetry"]["spans"]
        self.assertEqual(spans["scribnic.workflow.run"]["count"], 2)
