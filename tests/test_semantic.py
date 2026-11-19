"""Model metrics without models: settings, splitting, weighting, languages, extras."""

import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy

from scribe_eval.artifacts import Note, Segment, Transcript
from scribe_eval.config import DatasetEntry, load_experiment
from scribe_eval.datasets.prepared import PreparedDataset, write_dataset
from scribe_eval.metrics import METRICS, configure_metrics
from scribe_eval.metrics.clinical_bertscore import (
    ClinicalBertScore,
    keep_terms,
    medical_words,
    split_by_reference,
)
from scribe_eval.metrics.semantic import NegBleurt, _regression_score
from scribe_eval.types import Sample, Score
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
    def test_phrases_are_one_term(self):
        terms = frozenset({"insulin glargine", "insulin"})
        self.assertEqual(
            keep_terms("take insulin glargine daily", terms, 2), ["insulin_glargine"]
        )

    def test_terms_and_digits_count(self):
        terms = frozenset({"chest pain", "paracetamol"})
        found = medical_words(
            "take 2 paracetamol for chest pain and rest", terms, 2, "en"
        )
        self.assertEqual(found, {"2", "paracetamol", "chest", "pain"})

    def test_spelled_numbers_count_but_not_their_connectors(self):
        found = medical_words("tomar treinta y uno días", frozenset(), 1, "es")
        self.assertEqual(found, {"treinta", "uno"})


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


class NegBleurtTests(unittest.TestCase):
    def plan(self, language: str):
        """Validate NegBLEURT and BLEU on a dataset in `language`, without extras."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "tiny"
            write_tiny_dataset(root, ["a"])
            write_dataset(
                root,
                name="tiny",
                version="1",
                language=language,
                license="test",
                source="test",
            )
            entry = DatasetEntry("tiny", "all", "audio", ("negbleurt", "bleu"))
            with patch("importlib.util.find_spec", return_value=None):
                return plan_dataset(entry, PreparedDataset(root), FakePipeline())

    def score(self, language: str = "en", reference: str = "no fever"):
        """Score with a fake model that reports 0.25 and a truncated pair."""
        predicted = prediction(note=Note("fever"))
        expected = Sample("s", "all", language, {"note": Note(reference)})
        with (
            patch.object(NegBleurt, "_model", return_value=(None, None, "cpu")),
            patch(
                "scribe_eval.metrics.semantic._regression_score",
                return_value=(0.25, True),
            ) as regression,
        ):
            score = METRICS["negbleurt"].score(predicted, expected)
        return score, regression

    def test_other_languages_are_not_applicable_with_a_reason(self):
        plan = self.plan("es")
        self.assertEqual([metric.name for metric in plan.metrics], ["bleu"])
        self.assertEqual(
            plan.not_applicable["negbleurt"], "the metric supports en, not es"
        )

    def test_english_needs_the_semantic_extra(self):
        with self.assertRaisesRegex(ValueError, "'negbleurt' needs the semantic"):
            self.plan("en")

    def test_checkpoint_comes_from_settings(self):
        metrics = configure_metrics(
            {"negbleurt": {"checkpoint": "local/negbleurt", "max_length": 128}},
            judge=None,
        )
        self.assertEqual(metrics["negbleurt"].checkpoint, "local/negbleurt")
        self.assertEqual(metrics["negbleurt"].max_length, 128)
        self.assertEqual(METRICS["negbleurt"].checkpoint, "tum-nlp/NegBLEURT")

    def test_reference_goes_first_and_truncation_is_recorded(self):
        score, regression = self.score()
        self.assertEqual(score, Score("negbleurt", 0.25, {"truncated": 1.0}))
        arguments = regression.call_args.kwargs
        self.assertEqual(arguments["reference"], "no fever")
        self.assertEqual(arguments["candidate"], "fever")
        self.assertEqual(arguments["max_length"], 512)

    def test_other_languages_score_nothing_and_load_nothing(self):
        score, regression = self.score(language="es")
        self.assertIsNone(score)
        regression.assert_not_called()

    def test_empty_reference_scores_nothing(self):
        score, _ = self.score(reference="  ")
        self.assertIsNone(score)

    def test_aggregate_averages_and_counts_truncated_pairs(self):
        scores = [
            Score("negbleurt", 0.2, {"truncated": 1.0}),
            Score("negbleurt", 0.6, {"truncated": 0.0}),
            Score("negbleurt", 0.4, {"truncated": 1.0}),
        ]
        result = METRICS["negbleurt"].aggregate(scores)
        self.assertAlmostEqual(result["mean"], 0.4)
        self.assertEqual(result["truncated"], 2)


class _FakeEncoding(dict):
    """Tokenizer output that stays where it is when moved to a device."""

    def to(self, device: str) -> "_FakeEncoding":
        """Return itself; the fake model needs no device."""
        return self


class _FakeTokenizer:
    """One token per word, plus three special tokens per pair, as in BERT."""

    def __init__(self) -> None:
        """Remember nothing until called."""
        self.batch_arguments: dict[str, object] = {}

    def __call__(self, reference, candidate, **arguments) -> _FakeEncoding:
        """Encode a pair; record the settings of the batched call."""
        if isinstance(reference, list):
            self.batch_arguments = arguments
            return _FakeEncoding()
        length = len(reference.split()) + len(candidate.split()) + 3
        return _FakeEncoding(input_ids=list(range(length)))


@unittest.skipUnless(find_spec("torch"), "needs the semantic extra")
class RegressionScoreTests(unittest.TestCase):
    def run_pair(self, *, max_length: int):
        """Score a six-token pair with a model that always outputs 0.5."""
        tokenizer = _FakeTokenizer()

        def model(**inputs):
            """Return a single regression output."""
            return SimpleNamespace(logits=numpy.array([[0.5]]))

        result = _regression_score(
            model,
            tokenizer,
            "cpu",
            reference="no fever",
            candidate="ok",
            max_length=max_length,
        )
        return result, tokenizer.batch_arguments

    def test_pair_within_the_limit_is_not_truncated(self):
        (value, truncated), _ = self.run_pair(max_length=6)
        self.assertEqual(value, 0.5)
        self.assertFalse(truncated)

    def test_pair_past_the_limit_is_truncated_and_flagged(self):
        (_, truncated), arguments = self.run_pair(max_length=5)
        self.assertTrue(truncated)
        self.assertIs(arguments["truncation"], True)
        self.assertEqual(arguments["max_length"], 5)


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
