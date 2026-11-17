"""Check that runs line up by metric, and that differences are flagged."""

import json
import tempfile
import unittest
from pathlib import Path

from scribe_eval.application import compare_runs
from scribe_eval.comparison import RunRecord, compare


def record(
    label: str,
    *,
    wer: float,
    sample_ids: tuple[str, ...] = ("a", "b"),
    wer_version: int = 1,
    state: str = "completed",
) -> RunRecord:
    """Build a run that scored WER on one dataset, and ruled out DER."""
    manifest = {
        "pipeline_type": "fake",
        "pipeline": {"name": label},
        "datasets": {
            "tiny": {
                "version": "1",
                "input": "audio",
                "sample_ids": list(sample_ids),
                "metrics": {"wer": wer_version},
            }
        },
    }
    results = {
        "datasets": {
            "tiny": {
                "n_selected": 2,
                "n_run": 2,
                "n_failed": 1,
                "failures": {"b": "boom"},
                "metrics": {"wer": {"n": 1, "global": wer}},
                "not_applicable": {"der": "no diarization"},
            }
        }
    }
    return RunRecord(label, manifest, {"state": state}, results)


class CompareTests(unittest.TestCase):
    def test_values_line_up_by_metric_and_statistic(self):
        comparison = compare([record("x", wer=0.2), record("y", wer=0.1)])
        tiny = comparison.datasets["tiny"]
        self.assertEqual(tiny.values[("wer", "global")], {"x": 0.2, "y": 0.1})
        self.assertEqual(tiny.coverage["y"]["n_failed"], 1)
        self.assertEqual(set(tiny.not_applicable["der"]), {"x", "y"})
        self.assertEqual(comparison.warnings, ())

    def test_differences_that_bias_a_comparison_are_warned(self):
        comparison = compare(
            [
                record("x", wer=0.2),
                record("y", wer=0.1, sample_ids=("a",), wer_version=2, state="failed"),
            ]
        )
        warnings = "\n".join(comparison.warnings)
        self.assertIn("y is failed", warnings)
        self.assertIn("different samples", warnings)
        self.assertIn("'wer' changed version", warnings)

    def test_needs_two_distinct_runs(self):
        with self.assertRaisesRegex(ValueError, "two runs"):
            compare([record("x", wer=0.2)])
        with self.assertRaisesRegex(ValueError, "repeat"):
            compare([record("x", wer=0.2), record("x", wer=0.1)])

    def test_unscored_run_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = []
            for label in ("x", "y"):
                run = record(label, wer=0.1)
                root = Path(directory) / label
                root.mkdir()
                (root / "manifest.json").write_text(json.dumps(run.manifest))
                (root / "status.json").write_text(json.dumps(run.status))
                runs.append(root)
            (runs[0] / "results.json").write_text(
                json.dumps(record("x", wer=0.1).results)
            )
            with self.assertRaisesRegex(ValueError, "y has no results"):
                compare_runs(runs)
            (runs[1] / "results.json").write_text(
                json.dumps(record("y", wer=0.3).results)
            )
            comparison = compare_runs(runs)
        self.assertEqual(
            comparison.datasets["tiny"].values[("wer", "global")], {"x": 0.1, "y": 0.3}
        )
