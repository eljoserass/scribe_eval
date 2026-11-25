"""Long-text strategies without models: windows, sentences, chunks, aggregation."""

import re
import unittest
from unittest.mock import patch

from scribe_eval.metrics.long_text import (
    best_match_mean,
    check_strategy,
    reference_chunks,
    smart_1,
    split_sentences,
    token_windows,
)


def fake_punkt(text: str, language: str) -> list[str]:
    """Split after each period, standing in for Punkt without its data."""
    return [part for part in re.split(r"(?<=\.)\s+", text) if part]


class WindowTests(unittest.TestCase):
    def test_a_text_that_fits_is_one_window(self):
        self.assertEqual(token_windows(50, size=510), [(0, 0, 50)])

    def test_windows_overlap_and_each_token_is_embedded_once(self):
        windows = token_windows(1200, size=510, overlap=100)
        self.assertEqual(windows, [(0, 0, 510), (410, 510, 920), (820, 920, 1200)])
        embedded = []
        for _, first, end in windows:
            embedded.extend(range(first, end))
        self.assertEqual(embedded, list(range(1200)))
        for start, _, end in windows:
            self.assertLessEqual(end - start, 510)

    def test_window_must_be_larger_than_its_overlap(self):
        with self.assertRaisesRegex(ValueError, "can't overlap"):
            token_windows(300, size=100, overlap=100)


class SentenceTests(unittest.TestCase):
    def split(self, text: str, language: str = "en") -> tuple[list[str], list[str]]:
        """Split with a fake Punkt; return the sentences and languages it got."""
        languages = []

        def punkt(line: str, language: str) -> list[str]:
            """Record the language and split after periods."""
            languages.append(language)
            return fake_punkt(line, language)

        with (
            patch("scribe_eval.metrics.long_text._ensure_punkt"),
            patch("scribe_eval.metrics.long_text.sent_tokenize", side_effect=punkt),
        ):
            sentences = split_sentences(text, language, metric="bleurt")
        return sentences, languages

    def test_lines_are_units_before_punkt_splits_them(self):
        sentences, _ = self.split(
            "Plan:\n- Paracetamol 1 g. Review in 2 weeks.\n\n- No driving"
        )
        self.assertEqual(
            sentences,
            ["Plan:", "- Paracetamol 1 g.", "Review in 2 weeks.", "- No driving"],
        )

    def test_the_dataset_language_picks_the_punkt_model(self):
        _, languages = self.split("Fiebre. Tos.", "es")
        self.assertEqual(languages, ["spanish"])

    def test_a_language_without_punkt_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no model for 'eu'.*truncate"):
            self.split("Sukarra.", "eu")


class ChunkTests(unittest.TestCase):
    def test_a_short_reference_is_one_chunk(self):
        self.assertEqual(
            reference_chunks(["No fever.", "Cough."]), ["No fever. Cough."]
        )

    def test_chunks_hold_whole_sentences_as_alignscore_groups_them(self):
        sentences = [f"s{index} " + "word " * 99 for index in range(10)]
        chunks = reference_chunks(sentences, chunk_words=350)
        # 1,000 words ask for 3 chunks, so 3 sentences each and 1 left over.
        self.assertEqual(len(chunks), 4)
        self.assertTrue(chunks[0].startswith("s0 "))
        self.assertIn("s2 ", chunks[0])
        self.assertTrue(chunks[3].startswith("s9 "))


class AggregationTests(unittest.TestCase):
    def test_smart_1_matches_both_directions(self):
        # Rows are reference sentences, columns generated ones.
        precision, recall, f1 = smart_1([[1.0, 0.0], [0.0, 0.0], [0.0, 0.5]])
        self.assertAlmostEqual(precision, 0.75)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.6)

    def test_smart_1_is_zero_when_nothing_matches(self):
        self.assertEqual(smart_1([[0.0]]), (0.0, 0.0, 0.0))

    def test_best_match_mean_keeps_each_sentence_best_chunk(self):
        self.assertAlmostEqual(best_match_mean([[0.2, 0.9], [0.6, 0.1]]), 0.75)


class StrategyCheckTests(unittest.TestCase):
    def test_a_strategy_the_metric_does_not_take_is_named_in_the_error(self):
        with self.assertRaisesRegex(
            ValueError, "'bleurt' takes long_text = 'truncate', 'sentences'.*'window'"
        ):
            check_strategy("bleurt", "window", ("truncate", "sentences"))


if __name__ == "__main__":
    unittest.main()
