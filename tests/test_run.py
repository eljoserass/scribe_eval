"""Run whole experiments on a fake pipeline: records, failures, and resume."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scribe_eval import create_pipeline, resume_run, run_experiment, score_run
from scribe_eval.storage import append_jsonl, read_jsonl
from scribe_eval.telemetry.otlp import summarize

from .helpers import EXPERIMENT, FakePipeline, write_tiny_dataset


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

    def run_it(self, pipeline=None):
        """Run the experiment."""
        return run_experiment(
            self.experiment,
            data_directory=self.data,
            run_directory=self.run_directory,
            pipeline=pipeline,
        )

    def resume(self, pipeline, **settings):
        """Resume the run."""
        return resume_run(
            self.run_directory,
            data_directory=self.data,
            pipeline=pipeline,
            **settings,
        )

    def rows(self, name: str) -> list[dict]:
        """Read a dataset's JSONL file."""
        return read_jsonl(self.run_directory / "datasets" / "tiny" / f"{name}.jsonl")

    def json(self, name: str) -> dict:
        """Read one of the run's JSON files."""
        return json.loads((self.run_directory / name).read_text("utf-8"))

    def test_the_pipeline_comes_from_the_config(self):
        _, results = self.run_it()
        self.assertEqual(results["datasets"]["tiny"]["n_run"], 3)
        self.assertEqual(self.json("manifest.json")["pipeline"]["name"], "fake")

    def test_clean_run_records_everything(self):
        _, results = self.run_it(FakePipeline())
        dataset = results["datasets"]["tiny"]
        self.assertEqual(dataset["n_failed"], 0)
        self.assertEqual(dataset["metrics"]["wer"]["global"], 0.0)
        self.assertEqual(dataset["metrics"]["soap_sections"]["rate"], 1.0)
        self.assertEqual(self.json("status.json")["state"], "completed")
        manifest = self.json("manifest.json")
        self.assertEqual(manifest["datasets"]["tiny"]["sample_ids"], ["a", "b", "c"])
        self.assertIn("accelerators", manifest["hardware"])

    def test_the_pipeline_stays_open_by_default(self):
        pipeline = FakePipeline()
        self.run_it(pipeline)
        self.assertEqual(pipeline.opens, 1)
        self.assertEqual(self.json("manifest.json")["isolate_samples"], False)

    def test_isolated_samples_reopen_the_pipeline(self):
        text = EXPERIMENT.replace(
            'name = "tiny"\n', 'name = "tiny"\nisolate_samples = true\n', 1
        )
        self.experiment.write_text(text, encoding="utf-8")
        pipeline = FakePipeline()
        self.run_it(pipeline)
        self.assertEqual(pipeline.opens, 3)
        folders = {row["telemetry_directory"] for row in self.rows("predictions")}
        self.assertEqual(len(folders), 3)

    def test_the_harness_times_samples_from_the_outside(self):
        self.run_it(FakePipeline())
        self.assertGreater(self.rows("predictions")[0]["telemetry"]["wall_seconds"], 0)

    def test_report_is_written_and_cannot_break_out_of_its_script(self):
        self.run_it(FakePipeline())
        page = (self.run_directory / "report.html").read_text("utf-8")
        self.assertIn('"name": "tiny"', page)
        self.assertEqual(page.count("</script>"), 2)

    def test_failed_sample_is_a_row_and_keeps_partial_artifacts(self):
        _, results = self.run_it(FakePipeline(fail_notes_for={"b"}))
        dataset = results["datasets"]["tiny"]
        self.assertEqual(dataset["n_failed"], 1)
        self.assertIn("Note rejected", dataset["failures"]["b"])
        self.assertEqual(dataset["metrics"]["wer"]["n"], 3)
        self.assertEqual(dataset["metrics"]["soap_sections"]["n"], 2)
        self.assertEqual(dataset["metrics"]["elapsed_seconds"]["n"], 2)

    def test_interrupted_run_resumes_without_repeating_samples(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_it(FakePipeline(interrupt_after=1))
        self.assertEqual(self.json("status.json")["state"], "interrupted")
        self.assertEqual(len(self.rows("predictions")), 1)

        pipeline = FakePipeline()
        results = self.resume(pipeline)
        self.assertEqual(pipeline.runs, 2)
        self.assertEqual(results["datasets"]["tiny"]["n_run"], 3)
        self.assertEqual(self.json("status.json")["state"], "completed")
        attempts = self.run_directory / "datasets/tiny/samples/b"
        self.assertEqual(
            sorted(path.name for path in attempts.iterdir()),
            ["attempt-1", "attempt-2"],
        )

    def test_errors_are_retried_only_on_request(self):
        self.run_it(FakePipeline(fail_notes_for={"b"}))
        pipeline = FakePipeline()
        self.resume(pipeline)
        self.assertEqual(pipeline.runs, 0)

        results = self.resume(pipeline, retry_errors=True)
        self.assertEqual(pipeline.runs, 1)
        self.assertEqual(results["datasets"]["tiny"]["n_failed"], 0)
        self.assertEqual(len(self.rows("predictions")), 4)

    def test_resume_refuses_a_changed_pipeline(self):
        self.run_it(FakePipeline())
        with self.assertRaisesRegex(ValueError, "pipeline changed"):
            self.resume(FakePipeline(version=2))

    def test_scoring_runs_no_pipeline(self):
        self.run_it(FakePipeline())
        pipeline = FakePipeline()
        results = score_run(
            self.run_directory, data_directory=self.data, pipeline=pipeline
        )
        self.assertEqual(pipeline.runs, 0)
        self.assertEqual(results["datasets"]["tiny"]["metrics"]["wer"]["n"], 3)

    def test_a_run_is_only_completed_after_scoring(self):
        states = []

        class Failing(FakePipeline):
            """Record the status when scoring asks for a timeline."""

            def timeline(_, directories, telemetry):
                """Read the status as the report is written."""
                states.append(self.json("status.json")["state"])
                return super().timeline(directories, telemetry)

        self.run_it(Failing())
        self.assertEqual(set(states), {"completed"})

    def test_invalid_experiment_writes_nothing(self):
        text = EXPERIMENT.replace('input = "audio"', 'input = "note"')
        self.experiment.write_text(text, encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_it(FakePipeline())
        self.assertFalse(self.run_directory.exists())


class FactoryTests(unittest.TestCase):
    def test_unknown_types_explain_the_options(self):
        with self.assertRaisesRegex(ValueError, "package.module:ClassName"):
            create_pipeline("nope", {}, base_directory=Path("."))

    def test_missing_packages_are_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "installed"):
            create_pipeline("not_a_package.x:Y", {}, base_directory=Path("."))

    def test_any_class_can_be_named_in_the_config(self):
        pipeline = create_pipeline(
            "tests.helpers:FakePipeline", {"version": 7}, base_directory=Path(".")
        )
        self.assertEqual(pipeline.describe()["version"], 7)


class IndependenceTests(unittest.TestCase):
    def test_the_harness_never_loads_scribnic_on_its_own(self):
        code = (
            "import sys, scribe_eval, scribe_eval.cli, scribe_eval.application;"
            "assert 'scribnic' not in sys.modules, 'scribnic was imported'"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


class StorageTests(unittest.TestCase):
    def test_a_cut_off_last_line_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            append_jsonl(path, {"a": 1})
            with path.open("a", encoding="utf-8") as stream:
                stream.write('{"a": ')
            self.assertEqual(read_jsonl(path), [{"a": 1}])


class OtlpTests(unittest.TestCase):
    def test_machine_memory_reads_the_used_state_of_the_gauge(self):
        def point(state: str, value: str) -> dict:
            """Build a data point."""
            attribute = {"key": "state", "value": {"stringValue": state}}
            return {"attributes": [attribute], "timeUnixNano": "1", "asInt": value}

        points = [point("used", "42"), point("free", "99")]
        metric = {"name": "system.memory.usage", "gauge": {"dataPoints": points}}
        row = {"resourceMetrics": [{"scopeMetrics": [{"metrics": [metric]}]}]}
        with tempfile.TemporaryDirectory() as directory:
            append_jsonl(Path(directory) / "execution-1" / "metrics.jsonl", row)
            summary = summarize(Path(directory))
        self.assertEqual(summary["machine_memory_bytes"]["peak"], 42)
