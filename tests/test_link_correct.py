"""Check LINK and CORRECT against ADS_evaluation's formulas, with no model."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scribe_eval.artifacts import Note
from scribe_eval.judge import Judge, JudgeConfig
from scribe_eval.metrics import METRICS, link_correct
from scribe_eval.telemetry import SampleTelemetry
from scribe_eval.types import Prediction, Sample

PROMPTS = ("Extract: {input}", "Facts: {reference_facts} Check: {pred_fact}")


class ScriptedJudge:
    """Know each note's facts, and which generated facts are supported."""

    def __init__(self, facts: dict[str, list[str]], supported: set[str]):
        """Keep the answers."""
        self.facts = facts
        self.supported = supported

    def ask(self, prompt: str, schema: dict) -> dict:
        """Answer an extraction or a verdict."""
        if prompt.startswith("Extract"):
            facts = self.facts[prompt.removeprefix("Extract: ")]
            return {
                "facts": [
                    {"fact_id": index, "content": fact}
                    for index, fact in enumerate(facts)
                ]
            }
        fact = prompt.split("Check: ")[1]
        answer = "YES" if fact in self.supported else "NO"
        return {"supported": answer, "supporting_claim": "", "explanation": ""}


def score(name: str, judge: ScriptedJudge):
    """Score a generated note against a reference note."""
    prediction = Prediction(
        "d",
        "s",
        "clean",
        0,
        "audio",
        "success",
        None,
        {"note": Note("generated")},
        SampleTelemetry(wall_seconds=1.0),
        "x",
        "t",
    )
    sample = Sample("s", "all", "en", {"note": Note("reference")})
    with mock.patch.object(link_correct, "_prompts", return_value=PROMPTS):
        return METRICS[name].with_judge(judge).score(prediction, sample)


class FormulaTests(unittest.TestCase):
    def test_correct_is_supported_over_generated(self):
        facts = {"generated": ["a", "b", "c", "d"], "reference": ["a", "b"]}
        self.assertEqual(score("correct", ScriptedJudge(facts, {"a"})).value, 0.25)

    def test_link_is_supported_over_reference(self):
        facts = {"generated": ["a", "b", "c", "d"], "reference": ["a", "b"]}
        self.assertEqual(score("link", ScriptedJudge(facts, {"a"})).value, 0.5)

    def test_link_is_capped_at_one_as_in_their_code(self):
        facts = {"generated": ["a", "b", "c"], "reference": ["x"]}
        judge = ScriptedJudge(facts, {"a", "b", "c"})
        self.assertEqual(score("link", judge).value, 1.0)

    def test_a_note_without_facts_has_no_score(self):
        facts = {"generated": [], "reference": ["x"]}
        self.assertIsNone(score("correct", ScriptedJudge(facts, set())))


class PromptTests(unittest.TestCase):
    def test_finds_class_and_instance_prompts(self):
        source = 'class A:\n    EVAL = "x"\n    def f(self):\n        self.FACT = "y"\n'
        self.assertEqual(link_correct.prompt_from_source(source, "EVAL"), "x")
        self.assertEqual(link_correct.prompt_from_source(source, "FACT"), "y")

    def test_a_missing_prompt_is_an_error(self):
        with self.assertRaises(ValueError):
            link_correct.prompt_from_source("a = 1", "EVAL")


class CountingJudge(Judge):
    """Replace the HTTP call."""

    calls = 0

    def _request(self, prompt: str, schema: dict) -> dict:
        """Count calls instead of reaching a server."""
        type(self).calls += 1
        return {"answer": prompt}


class JudgeTests(unittest.TestCase):
    def test_remote_judges_need_permission(self):
        with self.assertRaisesRegex(ValueError, "not local"):
            JudgeConfig("https://api.example.com/v1", "m")
        JudgeConfig("https://api.example.com/v1", "m", allow_remote=True)
        JudgeConfig("http://127.0.0.1:8080/v1", "m")

    def test_answers_survive_in_the_cache(self):
        config = JudgeConfig("http://localhost/v1", "m")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.jsonl"
            CountingJudge.calls = 0
            first = CountingJudge(config, path)
            self.assertEqual(first.ask("q", {}), {"answer": "q"})
            first.ask("q", {})
            self.assertEqual(CountingJudge.calls, 1)
            CountingJudge(config, path).ask("q", {})
            self.assertEqual(CountingJudge.calls, 1)
