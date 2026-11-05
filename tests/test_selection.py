"""Check sample selection and experiment parsing."""

import tempfile
import unittest
from pathlib import Path

from scribe_eval.config import DatasetEntry, load_experiment
from scribe_eval.selection import select_ids

IDS = [f"s{number}" for number in range(10)]


def entry(**settings) -> DatasetEntry:
    """Build a dataset entry."""
    return DatasetEntry("d", "all", "audio", ("wer",), **settings)


class SelectionTests(unittest.TestCase):
    def test_everything_by_default(self):
        self.assertEqual(select_ids(entry(), IDS), tuple(sorted(IDS)))

    def test_same_seed_gives_the_same_subset(self):
        self.assertEqual(
            select_ids(entry(limit=3), IDS), select_ids(entry(limit=3), IDS)
        )
        self.assertEqual(len(select_ids(entry(limit=3), IDS)), 3)

    def test_fraction_rounds_up(self):
        self.assertEqual(len(select_ids(entry(fraction=0.25), IDS)), 3)

    def test_unknown_ids_fail(self):
        with self.assertRaises(ValueError):
            select_ids(entry(ids=("nope",)), IDS)


class ConfigTests(unittest.TestCase):
    def load(self, text: str):
        """Parse an experiment from text."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e.toml"
            path.write_text(text, encoding="utf-8")
            return load_experiment(path)

    def test_typos_fail(self):
        with self.assertRaisesRegex(ValueError, "worklow"):
            self.load('name = "e"\nworklow = "x"\n[[datasets]]\nname = "d"\n')

    def test_only_one_selector(self):
        text = """
        name = "e"
        [pipeline]
        type = "x"
        [[datasets]]
        name = "d"
        split = "all"
        input = "audio"
        metrics = []
        limit = 1
        fraction = 0.5
        """
        with self.assertRaisesRegex(ValueError, "only one"):
            self.load(text)
