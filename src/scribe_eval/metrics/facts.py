"""Check a note against the consultation, one clinical fact at a time.

Two separate questions, because they fail differently: what the note states that the
consultation doesn't support, and what the consultation established that the note
leaves out. The conversation is the reference, since clinicians' own notes contain
errors and differ in style.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Literal

from ..artifacts import DiarizedTranscript
from ..judge import Asker
from ..types import Prediction, Requires, Sample, Score

_BATCH_SIZE = 20
_FACTS_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "negated": {"type": "boolean"},
                },
                "required": ["text", "negated"],
            },
        }
    },
    "required": ["facts"],
}
_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer"},
                    "supported": {"type": "boolean"},
                },
                "required": ["index", "supported"],
            },
        }
    },
    "required": ["results"],
}
_EXTRACT_PROMPT = """{kind}:
{text}

Extract the atomic clinical facts from the text above.
- One piece of clinical information per fact: symptoms, history, findings, results,
  diagnoses, medication, doses, advice, follow-up.
- Make each fact self-contained, and keep figures, units, and doses exactly.
- Keep what was ruled out as facts, such as "no fever", and set negated to true.
- {scope}
- Write facts in the language of the text."""
_NOTE_SCOPE = "Include everything the note states."
_CONSULTATION_SCOPE = (
    "Include only what was established. Skip greetings, small talk, and questions "
    "nobody answered."
)
_CHECK_PROMPT = """{kind}:
{text}

Facts:
{facts}

For each fact, decide whether the text above supports it. It is supported only if the
text states it or directly implies it. A different value, dose, or frequency, a flipped
negation, or an added inference is not supported. Answer every fact, by index."""


@dataclass(frozen=True)
class Fact:
    """One atomic clinical statement."""

    text: str
    negated: bool


def _extract(judge: Asker, text: str, kind: str, scope: str) -> list[Fact]:
    """List the facts in a text."""
    prompt = _EXTRACT_PROMPT.format(kind=kind, text=text, scope=scope)
    answer = judge.ask(prompt, _FACTS_SCHEMA)
    return [Fact(fact["text"], fact["negated"]) for fact in answer["facts"]]


def _check_batch(
    judge: Asker, facts: Sequence[Fact], text: str, kind: str
) -> list[bool]:
    """Ask about a few facts, requiring an answer for each one."""
    listed = "\n".join(f"{i}. {fact.text}" for i, fact in enumerate(facts))
    prompt = _CHECK_PROMPT.format(kind=kind, text=text, facts=listed)
    results = judge.ask(prompt, _CHECK_SCHEMA)["results"]
    verdicts = {result["index"]: result["supported"] for result in results}
    if set(verdicts) != set(range(len(facts))):
        raise ValueError("The judge didn't answer every fact.")
    return [verdicts[index] for index in range(len(facts))]


def _check(judge: Asker, facts: Sequence[Fact], text: str, kind: str) -> list[bool]:
    """Check facts against a text, in batches a small model can handle."""
    verdicts = []
    for start in range(0, len(facts), _BATCH_SIZE):
        batch = facts[start : start + _BATCH_SIZE]
        verdicts += _check_batch(judge, batch, text, kind)
    return verdicts


def consultation_text(sample: Sample) -> str:
    """Render the reference conversation, with speakers when known."""
    transcript = sample.artifacts.get("diarized_transcript")
    if isinstance(transcript, DiarizedTranscript):
        return "\n".join(
            f"{segment.role or segment.speaker or 'unknown'}: {segment.text}"
            for segment in transcript.segments
        )
    return sample.artifact("transcript").text


@dataclass(frozen=True)
class FactRate:
    """Share of facts that fail, in one direction. Needs a judge to run."""

    name: str
    direction: Literal["unsupported", "omitted"]
    group: str = "note_facts"
    version: int = 1
    requires: Requires = Requires(("note",), ("transcript",))
    judge: Asker | None = None

    def with_judge(self, judge: Asker) -> "FactRate":
        """Return a copy that can score."""
        return replace(self, judge=judge)

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Count failing facts; a side with no facts has no rate."""
        note = prediction.artifacts["note"].text
        consultation = consultation_text(sample)
        if self.direction == "unsupported":
            facts = _extract(self.judge, note, "Clinical note", _NOTE_SCOPE)
            supported = _check(self.judge, facts, consultation, "Consultation")
        else:
            facts = _extract(
                self.judge, consultation, "Consultation", _CONSULTATION_SCOPE
            )
            supported = _check(self.judge, facts, note, "Clinical note")
        if not facts:
            return None
        failed = [f for f, ok in zip(facts, supported, strict=True) if not ok]
        negated = sum(fact.negated for fact in facts)
        failed_negated = sum(fact.negated for fact in failed)
        details = {
            "facts": float(len(facts)),
            "failed": float(len(failed)),
            "negated": float(negated),
            "failed_negated": float(failed_negated),
        }
        return Score(self.name, len(failed) / len(facts), details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the rate overall and, separately, among negations."""
        facts = sum(score.details["facts"] for score in scores)
        failed = sum(score.details["failed"] for score in scores)
        negated = sum(score.details["negated"] for score in scores)
        result = {
            "global": failed / facts,
            "sample_mean": fmean(score.value for score in scores),
            "facts_per_sample": facts / len(scores),
        }
        if negated:
            failed_negated = sum(score.details["failed_negated"] for score in scores)
            result["negations"] = failed_negated / negated
        return result
