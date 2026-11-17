"""Check grounding against the transcript with a scripted judge."""

import unittest

from scribe_eval.artifacts import DiarizedTranscript, Note, Segment, Transcript
from scribe_eval.metrics import METRICS
from scribe_eval.telemetry import SampleTelemetry
from scribe_eval.types import Prediction, Sample, Score


class ScriptedJudge:
    """Return fixed facts, and support the ones whose text the prompt contains."""

    def __init__(self, facts: list[str], supported: set[str]):
        """Keep the answers and record every prompt."""
        self.facts = facts
        self.supported = supported
        self.prompts: list[str] = []

    def ask(self, prompt: str, schema: dict) -> dict:
        """Answer an extraction or a verdict, telling them apart by schema."""
        self.prompts.append(prompt)
        if "facts" in schema["properties"]:
            return {"facts": self.facts}
        fact = next(fact for fact in self.facts if fact.strip() and fact in prompt)
        answer = "YES" if fact in self.supported else "NO"
        return {"explanation": "", "supported": answer}


def prediction(note: str) -> Prediction:
    """Wrap a generated note."""
    return Prediction(
        "d",
        "s",
        "clean",
        0,
        "audio",
        "success",
        None,
        {"note": Note(note)},
        SampleTelemetry(wall_seconds=1.0),
        "x",
        "t",
    )


def sample(transcript, language: str = "es") -> Sample:
    """A sample whose only reference is a transcript."""
    return Sample("s", "all", language, {transcript.kind: transcript})


DIARIZED = DiarizedTranscript(
    (Segment("¿Le duele el pecho?", "Doctor"), Segment("No.", "Paciente"))
)


class GroundingTests(unittest.TestCase):
    def test_score_is_supported_over_extracted_facts(self):
        judge = ScriptedJudge(
            ["niega dolor torácico", "toma 40 mg", " "], {"niega dolor torácico"}
        )
        score = (
            METRICS["grounding"]
            .with_judge(judge)
            .score(prediction("nota"), sample(DIARIZED))
        )
        self.assertEqual(score.value, 0.5)
        self.assertEqual(score.details, {"supported": 1.0, "facts": 2.0})

    def test_verdict_prompt_shows_speakers_and_puts_transcript_first(self):
        judge = ScriptedJudge(["niega dolor torácico"], set())
        METRICS["grounding"].with_judge(judge).score(
            prediction("nota"), sample(DIARIZED)
        )
        verdict = judge.prompts[1]
        self.assertIn("Paciente: No.", verdict)
        self.assertLess(verdict.index("Doctor:"), verdict.index("niega dolor torácico"))

    def test_plain_transcript_works_in_english(self):
        judge = ScriptedJudge(["no chest pain"], {"no chest pain"})
        score = (
            METRICS["grounding"]
            .with_judge(judge)
            .score(
                prediction("note"), sample(Transcript.from_text("No chest pain."), "en")
            )
        )
        self.assertEqual(score.value, 1.0)
        self.assertIn("Transcript of a medical consultation", judge.prompts[1])

    def test_note_without_facts_gets_no_score(self):
        judge = ScriptedJudge([], set())
        score = (
            METRICS["grounding"]
            .with_judge(judge)
            .score(prediction("S: No consta"), sample(DIARIZED))
        )
        self.assertIsNone(score)

    def test_unsupported_language_fails_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "'de'"):
            METRICS["grounding"].describe("de")

    def test_aggregate_reports_mean_and_pooled_share(self):
        scores = [
            Score("grounding", 1.0, {"supported": 1.0, "facts": 1.0}),
            Score("grounding", 0.5, {"supported": 3.0, "facts": 6.0}),
        ]
        aggregates = METRICS["grounding"].aggregate(scores)
        self.assertEqual(aggregates["mean"], 0.75)
        self.assertAlmostEqual(aggregates["pooled"], 4 / 7)
