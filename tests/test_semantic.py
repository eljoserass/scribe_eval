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
from scribe_eval.metrics.semantic import (
    BertScore,
    Bleurt,
    NegBleurt,
    _regression_scores,
    _window_embedding,
    _windowed_bertscore,
)
from scribe_eval.types import Sample, Score
from scribe_eval.validation import plan_dataset

from .helpers import FakePipeline, write_tiny_dataset
from .test_long_text import fake_punkt
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
                "scribe_eval.metrics.semantic._regression_scores",
                return_value=([0.25], 1),
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
        model, tokenizer, device, references, candidates = regression.call_args.args
        self.assertEqual((references, candidates), (["no fever"], ["fever"]))
        self.assertEqual(regression.call_args.kwargs["max_length"], 512)

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
        self.padded_calls: list[dict[str, object]] = []

    def __call__(self, references, candidates, **arguments) -> _FakeEncoding:
        """Encode pairs; record the settings of calls that pad for the model."""
        ids = [
            list(range(len(reference.split()) + len(candidate.split()) + 3))
            for reference, candidate in zip(references, candidates, strict=True)
        ]
        if "padding" in arguments:
            self.padded_calls.append(arguments)
        return _FakeEncoding(input_ids=ids)


@unittest.skipUnless(find_spec("torch"), "needs the semantic extra")
class RegressionScoreTests(unittest.TestCase):
    def run_pairs(self, *, max_length: int):
        """Score a six- and a seven-token pair, one per batch."""
        tokenizer = _FakeTokenizer()

        def model(input_ids):
            """Output a tenth of each pair's length, to check the order."""
            return SimpleNamespace(
                logits=numpy.array([[len(ids) / 10] for ids in input_ids])
            )

        result = _regression_scores(
            model,
            tokenizer,
            "cpu",
            ["no fever", "cough"],
            ["ok", "a dry cough"],
            max_length=max_length,
            batch_size=1,
        )
        return result, tokenizer.padded_calls

    def test_pairs_within_the_limit_are_scored_in_order(self):
        (values, truncated), calls = self.run_pairs(max_length=7)
        self.assertEqual(values, [0.6, 0.7])
        self.assertEqual(truncated, 0)
        self.assertEqual(len(calls), 2)

    def test_pairs_past_the_limit_are_truncated_and_counted(self):
        (_, truncated), calls = self.run_pairs(max_length=6)
        self.assertEqual(truncated, 1)
        self.assertIs(calls[0]["truncation"], True)
        self.assertEqual(calls[0]["max_length"], 6)


class LongTextSettingsTests(unittest.TestCase):
    def test_every_metric_truncates_by_default(self):
        for name in ("bertscore", "bleurt", "negbleurt"):
            self.assertEqual(METRICS[name].long_text, "truncate")

    def test_each_family_takes_its_own_strategies(self):
        metrics = configure_metrics(
            {
                "bertscore": {"long_text": "window"},
                "bleurt": {"long_text": "chunks"},
                "negbleurt": {"long_text": "sentences"},
            },
            judge=None,
        )
        self.assertEqual(metrics["bertscore"].long_text, "window")
        self.assertEqual(metrics["bleurt"].long_text, "chunks")
        self.assertEqual(metrics["negbleurt"].long_text, "sentences")

    def test_bertscore_rejects_sentences(self):
        with self.assertRaisesRegex(
            ValueError, "'bertscore' takes long_text = 'truncate', 'window'"
        ):
            configure_metrics({"bertscore": {"long_text": "sentences"}}, judge=None)

    def test_bleurt_rejects_windows(self):
        with self.assertRaisesRegex(ValueError, "not 'window'"):
            configure_metrics({"bleurt": {"long_text": "window"}}, judge=None)

    def test_negbleurt_rejects_chunks(self):
        with self.assertRaisesRegex(
            ValueError, "'negbleurt' takes long_text = 'truncate', 'sentences'"
        ):
            configure_metrics({"negbleurt": {"long_text": "chunks"}}, judge=None)

    def test_clinical_bertscore_takes_no_strategy(self):
        with self.assertRaisesRegex(ValueError, "Unknown settings.*long_text"):
            configure_metrics(
                {"clinical_bertscore": {"long_text": "window"}}, judge=None
            )

    def describe(self, metric, language: str = "en") -> dict[str, object]:
        """Describe a metric without the semantic extra installed."""
        with (
            patch("scribe_eval.metrics.semantic.require_semantic"),
            patch("scribe_eval.metrics.semantic.version", return_value="0"),
        ):
            return metric.describe(language)

    def test_the_manifest_records_the_strategy_and_its_parameters(self):
        chunks = self.describe(Bleurt(long_text="chunks"))
        self.assertEqual(chunks["long_text"], "chunks")
        self.assertEqual(chunks["punkt"], "english")
        self.assertEqual(chunks["chunk_words"], 350)
        truncate = self.describe(NegBleurt())
        self.assertEqual(truncate["long_text"], "truncate")
        self.assertNotIn("punkt", truncate)

    def test_splitting_a_language_without_punkt_fails_validation(self):
        with self.assertRaisesRegex(ValueError, "no model for 'eu'"):
            self.describe(Bleurt(long_text="sentences"), "eu")


class BleurtStrategyTests(unittest.TestCase):
    def score(
        self, long_text: str, *, truncated: int = 0, candidate: str = "Fever. Cough."
    ):
        """Score with a fake model: 1 when the generated sentence is in the piece."""
        calls = []

        def fake_scores(model, tokenizer, device, references, candidates, **_):
            """Record the pairs and score them."""
            calls.append((list(references), list(candidates)))
            values = [
                float(candidate in reference)
                for reference, candidate in zip(references, candidates, strict=True)
            ]
            return values, truncated

        metric = Bleurt(long_text=long_text)
        with (
            patch.object(Bleurt, "_model", return_value=(None, None, "cpu")),
            patch(
                "scribe_eval.metrics.semantic._regression_scores",
                side_effect=fake_scores,
            ),
            patch("scribe_eval.metrics.long_text._ensure_punkt"),
            patch(
                "scribe_eval.metrics.long_text.sent_tokenize", side_effect=fake_punkt
            ),
        ):
            score = metric.score(
                prediction(note=Note(candidate)),
                sample(note=Note("Fever. Rash. Pain.")),
            )
        return score, calls

    def test_truncate_scores_the_notes_as_one_pair(self):
        score, calls = self.score("truncate")
        self.assertEqual(calls, [(["Fever. Rash. Pain."], ["Fever. Cough."])])
        self.assertEqual(score.details, {"truncated": 0.0})

    def test_sentences_pairs_every_sentence_and_reports_smart_1(self):
        score, calls = self.score("sentences")
        references, candidates = calls[0]
        self.assertEqual(references[:3], ["Fever.", "Fever.", "Rash."])
        self.assertEqual(candidates[:3], ["Fever.", "Cough.", "Fever."])
        self.assertAlmostEqual(score.details["precision"], 0.5)
        self.assertAlmostEqual(score.details["recall"], 1 / 3)
        self.assertAlmostEqual(score.value, 0.4)
        self.assertEqual(score.details["pairs"], 6)
        self.assertEqual(score.details["truncated"], 0)

    def test_chunks_keep_each_sentence_best_chunk(self):
        score, calls = self.score("chunks")
        self.assertEqual(calls[0][0], ["Fever. Rash. Pain.", "Fever. Rash. Pain."])
        self.assertAlmostEqual(score.value, 0.5)
        self.assertEqual(score.details, {"truncated": 0.0, "pairs": 2.0})

    def test_repeated_sentences_are_scored_once(self):
        score, calls = self.score("sentences", candidate="Fever. Cough. Fever.")
        references, candidates = calls[0]
        self.assertEqual(len(references), 6)
        self.assertEqual(score.details["pairs"], 9)
        self.assertAlmostEqual(score.details["precision"], 2 / 3)

    def test_a_truncated_pair_flags_the_note(self):
        score, _ = self.score("sentences", truncated=3)
        self.assertEqual(score.details["truncated"], 1.0)

    def test_aggregate_reports_precision_recall_and_pairs(self):
        score, _ = self.score("sentences")
        result = Bleurt(long_text="sentences").aggregate([score, score])
        self.assertAlmostEqual(result["mean"], 0.4)
        self.assertAlmostEqual(result["recall"], 1 / 3)
        self.assertEqual(result["pairs"], 12)
        self.assertEqual(result["truncated"], 0)


class _FakeScorer:
    """Stands in for BERTScorer, with fixed precision, recall, and F1."""

    def score(self, candidates, references, batch_size):
        """Return 0.9, 0.8, and 0.85 for the single pair."""
        return [0.9], [0.8], [0.85]


class BertScoreWindowTests(unittest.TestCase):
    def score(self, long_text: str, candidate: str):
        """Score against a two-word reference, with a limit of six tokens."""
        tokenizer = SimpleNamespace(
            model_max_length=6, num_special_tokens_to_add=lambda: 2
        )
        metric = BertScore(long_text=long_text)
        with (
            patch.object(BertScore, "_scorer", return_value=(_FakeScorer(), tokenizer)),
            patch(
                "scribe_eval.metrics.semantic._bertscore_ids",
                side_effect=lambda tokenizer, text: list(range(len(text.split()))),
            ),
            patch(
                "scribe_eval.metrics.semantic._windowed_bertscore",
                return_value=(0.7, 0.6, 0.65),
            ) as windowed,
        ):
            score = metric.score(
                prediction(note=Note(candidate)), sample(note=Note("no fever"))
            )
        return score, windowed

    def test_truncate_flags_a_long_note(self):
        score, windowed = self.score("truncate", "one two three four five")
        windowed.assert_not_called()
        self.assertEqual(score.value, 0.85)
        self.assertEqual(score.details["truncated"], 1.0)
        self.assertNotIn("windowed", score.details)

    def test_window_embeds_a_long_note_in_windows(self):
        score, windowed = self.score("window", "one two three four five")
        self.assertEqual(len(windowed.call_args.kwargs["candidate_ids"]), 5)
        self.assertEqual(score.value, 0.65)
        self.assertEqual(
            score.details,
            {"precision": 0.7, "recall": 0.6, "truncated": 0.0, "windowed": 1.0},
        )

    def test_window_leaves_a_pair_that_fits_to_the_package(self):
        score, windowed = self.score("window", "no fever today")
        windowed.assert_not_called()
        self.assertEqual(score.value, 0.85)
        self.assertEqual(score.details["windowed"], 0.0)

    def test_aggregate_counts_windowed_pairs(self):
        long_score, _ = self.score("window", "one two three four five")
        short_score, _ = self.score("window", "no fever today")
        result = BertScore(long_text="window").aggregate([long_score, short_score])
        self.assertEqual(result["windowed"], 1)
        self.assertEqual(result["truncated"], 0)


class _WindowTokenizer:
    """BERT-like: 112-token inputs, wrapped in IDs 101 and 102."""

    model_max_length = 112
    pad_token_id = 0
    cls_token_id = 101
    sep_token_id = 102

    def num_special_tokens_to_add(self) -> int:
        """Two special tokens per text."""
        return 2

    def build_inputs_with_special_tokens(self, ids: list[int]) -> list[int]:
        """Wrap the tokens."""
        return [101, *ids, 102]

    def get_special_tokens_mask(self, ids: list[int]) -> list[int]:
        """Mark the wrapping tokens."""
        return [1] + [0] * len(ids) + [1]


class _PositionModel:
    """Embeds each token as its ID and its position in the window."""

    def __init__(self) -> None:
        """Record batch sizes as they come."""
        self.batches: list[int] = []

    def eval(self) -> "_PositionModel":
        """Do nothing; there is no training mode."""
        return self

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        """Return embeddings of shape (batch, tokens, 2)."""
        import torch

        self.batches.append(input_ids.shape[0])
        positions = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        return (torch.stack([input_ids.float(), positions.float()], dim=-1),)


@unittest.skipUnless(find_spec("bert_score"), "needs the semantic extra")
class WindowEmbeddingTests(unittest.TestCase):
    def embed(self, length: int):
        """Embed `length` tokens in windows of 110 overlapping by 100."""
        model = _PositionModel()
        embedding, ids = _window_embedding(
            list(range(1000, 1000 + length)),
            model=model,
            tokenizer=_WindowTokenizer(),
            device="cpu",
            batch_size=2,
        )
        return embedding, ids, model

    def test_a_text_that_fits_is_one_input(self):
        embedding, ids, model = self.embed(10)
        self.assertEqual(ids, [101, *range(1000, 1010), 102])
        self.assertEqual(embedding[:, 1].tolist(), [float(i) for i in range(12)])
        self.assertEqual(model.batches, [1])

    def test_each_token_comes_once_from_the_first_window_holding_it(self):
        embedding, ids, model = self.embed(130)
        # Windows hold tokens 0-109, 10-119, and 20-129.
        self.assertEqual(ids, [101, *range(1000, 1130), 102])
        self.assertEqual(embedding[:, 0].tolist(), [float(i) for i in ids])
        self.assertEqual(embedding[1 + 50, 1].item(), 1 + 50)
        self.assertEqual(embedding[1 + 115, 1].item(), 1 + 115 - 10)
        self.assertEqual(embedding[-1, 1].item(), 1 + 130 - 20)
        self.assertEqual(model.batches, [2, 1])

    def test_identical_long_texts_match_fully(self):
        scorer = SimpleNamespace(
            _model=_PositionModel(), device="cpu", rescale_with_baseline=False
        )
        ids = list(range(1000, 1300))
        values = _windowed_bertscore(
            scorer,
            _WindowTokenizer(),
            candidate_ids=ids,
            reference_ids=ids,
            batch_size=4,
        )
        for value in values:
            self.assertAlmostEqual(value, 1.0, places=5)


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
