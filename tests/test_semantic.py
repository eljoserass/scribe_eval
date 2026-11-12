"""Model metrics without models: settings, splitting, weighting, missing extras."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scribe_eval.artifacts import Segment, Transcript
from scribe_eval.config import DatasetEntry, load_experiment
from scribe_eval.datasets.prepared import PreparedDataset
from scribe_eval.metrics import METRICS, configure_metrics
from scribe_eval.metrics.clinical_bertscore import (
    ClinicalBertScore,
    medical_words,
    split_by_reference,
)
from scribe_eval.validation import plan_dataset

from .helpers import FakePipeline, write_tiny_dataset
from .test_metrics import prediction, sample


class SplitTests(unittest.TestCase):
    def test_matches_and_substitutions_follow_their_reference_word(self):
        pairs = split_by_reference(
            ["i have a cough", "since monday"], "i had a cough since sunday"
        )
        self.assertEqual(
            pairs,
            [("i have a cough", "i had a cough"), ("since monday", "since sunday")],
        )

    def test_insertions_join_the_previous_utterance(self):
        pairs = split_by_reference(["no fever", "any rash"], "no fever at all any rash")
        self.assertEqual(pairs[0], ("no fever", "no fever at all"))

    def test_a_missed_utterance_gets_an_empty_hypothesis(self):
        pairs = split_by_reference(["no", "chest pain"], "chest pain")
        self.assertEqual(pairs, [("no", ""), ("chest pain", "chest pain")])

    def test_empty_utterances_are_dropped(self):
        self.assertEqual(split_by_reference(["", "ok"], "ok"), [("ok", "ok")])


class MedicalWordTests(unittest.TestCase):
    def test_terms_and_spelled_numbers_count(self):
        terms = frozenset({"chest pain", "paracetamol"})
        found = medical_words(
            "take two paracetamol for chest pain and rest", terms, 2, "en"
        )
        self.assertEqual(found, {"two", "paracetamol", "chest", "pain"})


class SettingsTests(unittest.TestCase):
    def test_settings_reach_the_metric(self):
        metrics = configure_metrics(
            {"bertscore": {"model": "distilbert-base-uncased", "device": "cpu"}},
            judge=None,
        )
        self.assertEqual(metrics["bertscore"].model, "distilbert-base-uncased")
        self.assertIsNone(METRICS["bertscore"].model)

    def test_unknown_setting_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown settings.*modle"):
            configure_metrics({"bertscore": {"modle": "x"}}, judge=None)

    def test_unknown_metric_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "unknown metrics: bertscroe"):
            configure_metrics({"bertscroe": {}}, judge=None)

    def test_metric_without_settings_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "takes no settings"):
            configure_metrics({"wer": {"x": 1}}, judge=None)

    def test_medical_weight_must_be_a_proportion(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            configure_metrics({"clinical_bertscore": {"medical_weight": 2}}, judge=None)

    def test_experiment_file_carries_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e.toml"
            path.write_text(
                'name = "e"\n[pipeline]\ntype = "fake"\n'
                '[[datasets]]\nname = "d"\nsplit = "all"\ninput = "audio"\nmetrics = []\n'
                '[metric_settings.bleurt]\ncheckpoint = "lucadiliello/BLEURT-20-D12"\n'
            )
            experiment = load_experiment(path)
        self.assertEqual(
            experiment.metric_settings,
            {"bleurt": {"checkpoint": "lucadiliello/BLEURT-20-D12"}},
        )


class ValidationTests(unittest.TestCase):
    def test_missing_extra_fails_before_any_model_loads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny"
            write_tiny_dataset(root, ["a"])
            entry = DatasetEntry("tiny", "all", "audio", ("note_semantic",))
            with patch("importlib.util.find_spec", return_value=None):
                with self.assertRaisesRegex(ValueError, "semantic extra"):
                    plan_dataset(entry, PreparedDataset(root), FakePipeline())


class ClinicalWeightingTests(unittest.TestCase):
    def score(self, reference_segments, predicted_text):
        """Score with fake BERTScores: 0.9 on all tokens, 0.5 on medical ones."""

        def fake_f1(encoder, pairs, words):
            return [0.9 if words is None else 0.5] * len(pairs)

        metric = ClinicalBertScore()
        segments = tuple(Segment(text) for text in reference_segments)
        reference = sample(transcript=Transcript(segments))
        predicted = prediction(transcript=Transcript.from_text(predicted_text))
        with (
            patch.object(ClinicalBertScore, "_encoder", return_value=None),
            patch.object(ClinicalBertScore, "_f1", side_effect=fake_f1),
        ):
            return metric.score(predicted, reference)

    def test_medical_utterances_mix_both_scores(self):
        score = self.score(["take ibuprofen"], "take ibuprofen")
        self.assertAlmostEqual(score.value, 0.4 * 0.5 + 0.6 * 0.9)
        self.assertEqual(score.details["medical_utterances"], 1)

    def test_utterances_without_medical_words_use_plain_bertscore(self):
        score = self.score(["hello there"], "hello there")
        self.assertAlmostEqual(score.value, 0.9)

    def test_sample_value_weighs_utterances_by_reference_words(self):
        score = self.score(
            ["take ibuprofen", "hello there my friend"],
            "take ibuprofen hello there my friend",
        )
        expected = (2 * (0.4 * 0.5 + 0.6 * 0.9) + 4 * 0.9) / 6
        self.assertAlmostEqual(score.value, expected)

    def test_languages_without_terms_do_not_apply(self):
        metric = ClinicalBertScore()
        spanish = sample(transcript=Transcript.from_text("hola"))
        spanish = type(spanish)("s", "all", "xx", spanish.artifacts, {})
        self.assertIsNone(
            metric.score(prediction(transcript=Transcript.from_text("hola")), spanish)
        )


if __name__ == "__main__":
    unittest.main()
