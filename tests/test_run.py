"""Run whole experiments on fake backends: records, failures, and resume."""

import json
import tempfile
import unittest
from pathlib import Path

from scribe_eval import resume_run, run_experiment, score_run
from scribe_eval.storage import append_jsonl, read_jsonl

from .helpers import EXPERIMENT, FakeBackend, fake_workflow, write_tiny_dataset


class Interrupt(FakeBackend):
    """Stop the run after a given number of notes."""

    def __init__(self, step, *, after: int, **settings):
        """Remember when to stop."""
        super().__init__(step, **settings)
        self.after = after

    def run(self, source):
        """Raise once enough calls happened."""
        if self.calls.count("note") >= self.after and self.step.name == "note":
            raise KeyboardInterrupt
        return super().run(source)


class RunTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.data = self.directory / "data"
        write_tiny_dataset(self.data / "tiny", ["a", "b", "c"])
        self.experiment = self.directory / "experiment.toml"
        self.experiment.write_text(EXPERIMENT, encoding="utf-8")
        self.run_directory = self.directory / "run"
        self.calls = []

    def factory(self, fail_notes_for=frozenset()):
        """Build fake backends sharing one call log."""

        def create(step):
            """Create one backend."""
            return FakeBackend(
                step, fail_notes_for=set(fail_notes_for), calls=self.calls
            )

        return create

    def run_it(self, factory):
        """Run the experiment with a fake workflow."""
        return run_experiment(
            self.experiment,
            data_directory=self.data,
            run_directory=self.run_directory,
            workflow=fake_workflow(),
            backend_factory=factory,
        )

    def rows(self, name: str) -> list[dict]:
        """Read a dataset's JSONL file."""
        return read_jsonl(self.run_directory / "datasets" / "tiny" / f"{name}.jsonl")

    def status(self) -> dict:
        """Read the run status."""
        return json.loads((self.run_directory / "status.json").read_text("utf-8"))

    def test_clean_run_records_everything(self):
        _, results = self.run_it(self.factory())
        dataset = results["datasets"]["tiny"]
        self.assertEqual(dataset["n_run"], 3)
        self.assertEqual(dataset["n_failed"], 0)
        self.assertEqual(dataset["metrics"]["wer"]["global"], 0.0)
        self.assertEqual(dataset["metrics"]["soap_sections"]["rate"], 1.0)
        self.assertEqual(self.status()["state"], "completed")
        manifest = json.loads((self.run_directory / "manifest.json").read_text("utf-8"))
        self.assertEqual(manifest["datasets"]["tiny"]["sample_ids"], ["a", "b", "c"])

    def test_input_is_not_recorded_as_a_prediction(self):
        self.run_it(self.factory())
        self.assertEqual(
            self.rows("predictions")[0]["artifacts"], ["note", "transcript"]
        )

    def test_failed_sample_is_a_row_and_keeps_partial_artifacts(self):
        _, results = self.run_it(self.factory({"b"}))
        dataset = results["datasets"]["tiny"]
        self.assertEqual(dataset["n_failed"], 1)
        self.assertIn("Note rejected", dataset["failures"]["b"])
        self.assertEqual(dataset["metrics"]["wer"]["n"], 3)
        self.assertEqual(dataset["metrics"]["soap_sections"]["n"], 2)
        self.assertEqual(dataset["metrics"]["elapsed_seconds"]["n"], 2)

    def test_interrupted_run_resumes_without_repeating_samples(self):
        def stopping(step):
            """Stop before the second note."""
            return Interrupt(step, after=1, fail_notes_for=set(), calls=self.calls)

        with self.assertRaises(KeyboardInterrupt):
            self.run_it(stopping)
        self.assertEqual(self.status()["state"], "interrupted")
        self.assertEqual(len(self.rows("predictions")), 1)

        self.calls.clear()
        results = resume_run(
            self.run_directory,
            data_directory=self.data,
            workflow=fake_workflow(),
            backend_factory=self.factory(),
        )
        self.assertEqual(self.calls.count("note"), 2)
        self.assertEqual(results["datasets"]["tiny"]["n_run"], 3)
        self.assertEqual(self.status()["state"], "completed")
        attempts = self.run_directory / "datasets/tiny/samples/b"
        self.assertEqual(
            sorted(path.name for path in attempts.iterdir()),
            ["attempt-1", "attempt-2"],
        )

    def test_errors_are_retried_only_on_request(self):
        self.run_it(self.factory({"b"}))
        self.calls.clear()
        arguments = {
            "data_directory": self.data,
            "workflow": fake_workflow(),
            "backend_factory": self.factory(),
        }
        resume_run(self.run_directory, **arguments)
        self.assertEqual(self.calls, [])

        results = resume_run(self.run_directory, retry_errors=True, **arguments)
        self.assertEqual(self.calls.count("note"), 1)
        self.assertEqual(results["datasets"]["tiny"]["n_failed"], 0)
        self.assertEqual(len(self.rows("predictions")), 4)

    def test_resume_refuses_a_changed_workflow(self):
        self.run_it(self.factory())
        changed = fake_workflow()
        changed = type(changed)("other", changed.steps, inputs=changed.inputs)
        with self.assertRaisesRegex(ValueError, "workflow changed"):
            resume_run(self.run_directory, data_directory=self.data, workflow=changed)

    def test_scoring_needs_no_model(self):
        self.run_it(self.factory())
        self.calls.clear()
        results = score_run(
            self.run_directory, data_directory=self.data, workflow=fake_workflow()
        )
        self.assertEqual(self.calls, [])
        self.assertEqual(results["datasets"]["tiny"]["metrics"]["wer"]["n"], 3)

    def test_invalid_experiment_writes_nothing(self):
        text = EXPERIMENT.replace('input = "audio"', 'input = "note"')
        self.experiment.write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_it(self.factory())
        self.assertFalse(self.run_directory.exists())


class StorageTests(unittest.TestCase):
    def test_a_cut_off_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            append_jsonl(path, {"a": 1})
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"a": ')
            self.assertEqual(read_jsonl(path), [{"a": 1}])
