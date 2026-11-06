"""Check fact metrics and the judge client, with no model behind them."""

import tempfile
import unittest
from pathlib import Path

from scribe_eval.artifacts import DiarizedTranscript, Note, Segment, Transcript
from scribe_eval.judge import Judge, JudgeConfig
from scribe_eval.metrics import METRICS
from scribe_eval.metrics.facts import consultation_text
from scribe_eval.telemetry import SampleTelemetry
from scribe_eval.types import Prediction, Sample


class ScriptedJudge:
    """Answer extraction and checks from fixed lists."""

    def __init__(self, facts: list[dict], verdicts: list[bool]):
        """Keep the answers."""
        self.facts = facts
        self.verdicts = verdicts

    def ask(self, prompt: str, schema: dict) -> dict:
        """Tell the two questions apart by their schema."""
        if "facts" in schema["properties"]:
            return {"facts": self.facts}
        return {
            "results": [
                {"index": index, "supported": verdict}
                for index, verdict in enumerate(self.verdicts)
            ]
        }


def prediction() -> Prediction:
    """Build a prediction with a note."""
    return Prediction(
        "d",
        "s",
        "clean",
        0,
        "audio",
        "success",
        None,
        {"note": Note("S — no fever")},
        SampleTelemetry(wall_seconds=1.0),
        "x",
        "t",
    )


def sample() -> Sample:
    """Build a sample with a consultation."""
    return Sample("s", "all", "en", {"transcript": Transcript.from_text("Hello")})


class FactRateTests(unittest.TestCase):
    def test_unsupported_rate_tracks_negations_apart(self):
        facts = [
            {"text": "no fever", "negated": True},
            {"text": "takes ibuprofen", "negated": False},
        ]
        metric = METRICS["unsupported_fact_rate"].with_judge(
            ScriptedJudge(facts, [False, True])
        )
        score = metric.score(prediction(), sample())
        self.assertEqual(score.value, 0.5)
        result = metric.aggregate([score])
        self.assertEqual(result["negations"], 1.0)

    def test_a_note_without_facts_has_no_rate(self):
        metric = METRICS["omitted_fact_rate"].with_judge(ScriptedJudge([], []))
        self.assertIsNone(metric.score(prediction(), sample()))

    def test_a_partial_answer_is_an_error_not_a_verdict(self):
        facts = [{"text": "a", "negated": False}, {"text": "b", "negated": False}]
        metric = METRICS["omitted_fact_rate"].with_judge(ScriptedJudge(facts, [True]))
        with self.assertRaisesRegex(ValueError, "every fact"):
            metric.score(prediction(), sample())

    def test_speakers_reach_the_judge(self):
        transcript = DiarizedTranscript((Segment("Hello", "S1", role="doctor"),))
        case = Sample("s", "all", "en", {"diarized_transcript": transcript})
        self.assertEqual(consultation_text(case), "doctor: Hello")


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
