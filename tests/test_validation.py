"""Check that bad combinations are caught before loading any model."""

import tempfile
import unittest
from pathlib import Path

from scribe_eval.artifacts import Note, Transcript
from scribe_eval.config import DatasetEntry
from scribe_eval.datasets.prepared import (
    PreparedDataset,
    sample_directory,
    write_dataset,
    write_sample,
)
from scribe_eval.validation import plan_dataset

from .helpers import FakePipeline, write_tiny_dataset


class ValidationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "tiny"
        write_tiny_dataset(self.root, ["a"])
        self.dataset = PreparedDataset(self.root)

    def plan(self, input_kind: str, metrics: tuple[str, ...]):
        """Validate one entry against the fake pipeline."""
        entry = DatasetEntry("tiny", "all", input_kind, metrics)
        return plan_dataset(entry, self.dataset, FakePipeline())

    def test_transcript_input_makes_wer_not_applicable(self):
        plan = self.plan("transcript", ("wer", "soap_sections"))
        self.assertEqual([metric.name for metric in plan.metrics], ["soap_sections"])
        self.assertIn("does not produce transcript", plan.not_applicable["wer"])

    def test_audio_input_allows_wer(self):
        self.assertEqual(self.plan("audio", ("wer",)).not_applicable, {})

    def test_missing_reference_is_an_error(self):
        root = self.root.parent / "no_notes"
        sample_directory(root, "a").mkdir(parents=True)
        (sample_directory(root, "a") / "audio.wav").touch()
        write_sample(root, "a", [Transcript.from_text("x")], {})
        write_dataset(
            root, name="no_notes", version="1", language="en", license="t", source="t"
        )
        entry = DatasetEntry("no_notes", "all", "audio", ("rouge_l",))
        with self.assertRaisesRegex(ValueError, "'note' reference"):
            plan_dataset(entry, PreparedDataset(root), FakePipeline())

    def test_missing_input_is_an_error(self):
        root = self.root.parent / "text_only"
        write_sample(root, "a", [Note("x")], {})
        write_dataset(
            root, name="text_only", version="1", language="en", license="t", source="t"
        )
        entry = DatasetEntry("text_only", "all", "audio", ())
        with self.assertRaisesRegex(ValueError, "no 'audio' input"):
            plan_dataset(entry, PreparedDataset(root), FakePipeline())

    def test_judged_metrics_need_a_judge(self):
        with self.assertRaisesRegex(ValueError, "judge"):
            self.plan("audio", ("link_correct",))

    def test_unprepared_dataset_explains_what_to_do(self):
        with self.assertRaisesRegex(ValueError, "scribe-eval prepare"):
            PreparedDataset(self.root.parent / "missing")
