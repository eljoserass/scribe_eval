"""Grounding: how much of a note the reference transcript supports.

A judge splits the generated note into atomic facts, then decides, fact by fact,
whether the reference transcript supports it, in the style of FActScore and DocLens
attribution:

    grounding = supported facts / generated facts

It needs no reference note, so it works on datasets with transcripts only, such as
TEME-v1. Unsupported facts are invented, or added by the transcription step; the
reference transcript can't tell those apart. The prompts are this project's, written
for this metric, and have not been validated against human judgments.

References:
    Min et al., "FActScore: Fine-grained Atomic Evaluation of Factual Precision in
    Long Form Text Generation", EMNLP 2023. https://arxiv.org/abs/2305.14251
    Xie et al., "DocLens: Multi-aspect Fine-grained Evaluation for Medical Text
    Generation", ACL 2024. https://arxiv.org/abs/2311.09581
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..artifacts import DiarizedTranscript
from ..judge import Asker
from ..types import Prediction, Requires, Sample, Score

_FACTS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {"type": "string"}}},
    "required": ["facts"],
}
# The explanation comes first so the judge reasons before it decides.
_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "supported": {"type": "string", "enum": ["YES", "NO"]},
    },
    "required": ["explanation", "supported"],
}

# Prompts by dataset language. The transcript leads the verdict prompt so a server's
# prompt cache can reuse it across the facts of one note.
_EXTRACT_PROMPTS = {
    "en": (
        "Split the following clinical note into atomic facts. Each fact is a short, "
        "self-contained statement that can be checked on its own: a symptom, a "
        "history item, a medication with its dose, a finding, or a plan. Keep "
        "negations, such as a denied symptom. Ignore section headings and phrases "
        "that only say information is missing. Add nothing the note does not say."
        "\n\nNote:\n{note}"
    ),
    "es": (
        "Divide la siguiente nota clínica en hechos atómicos. Cada hecho es una "
        "afirmación breve y autónoma que se puede comprobar por separado: un "
        "síntoma, un antecedente, un medicamento con su dosis, un hallazgo o un "
        "plan. Conserva las negaciones, como un síntoma que el paciente niega. "
        "Ignora los encabezados de sección y las frases que solo indican que falta "
        "información, como «No consta». No añadas nada que la nota no diga."
        "\n\nNota:\n{note}"
    ),
}
_VERDICT_PROMPTS = {
    "en": (
        "Transcript of a medical consultation:\n{transcript}\n\n"
        "Fact taken from a clinical note written from that consultation:\n{fact}\n\n"
        "Does the transcript support the fact? Answer YES if it is said explicitly "
        "or follows directly from what was said. Answer NO if it does not appear, "
        "contradicts the transcript, or adds details nobody mentioned, such as a "
        "dose, a date, a side, or a number. Explain in one sentence first."
    ),
    "es": (
        "Transcripción de una consulta médica:\n{transcript}\n\n"
        "Hecho tomado de una nota clínica redactada a partir de esa consulta:\n"
        "{fact}\n\n"
        "¿La transcripción respalda el hecho? Responde YES si se dice explícitamente "
        "o se deduce directamente de lo dicho. Responde NO si no aparece, contradice "
        "la transcripción o añade detalles que nadie mencionó, como una dosis, una "
        "fecha, un lado del cuerpo o una cifra. Explícalo antes en una frase."
    ),
}


def _transcript_text(sample: Sample) -> str:
    """Render the reference transcript, with speakers when the dataset has them.

    Speakers matter here: a symptom the doctor asks about is not one the patient has.
    """
    diarized = sample.artifacts.get("diarized_transcript")
    if not isinstance(diarized, DiarizedTranscript):
        return sample.artifact("transcript").text
    lines = []
    for segment in diarized.segments:
        speaker = segment.speaker or "?"
        lines.append(f"{speaker}: {segment.text}")
    return "\n".join(lines)


def _extract(judge: Asker, note: str, language: str) -> list[str]:
    """List a note's atomic facts, dropping empty ones."""
    prompt = _EXTRACT_PROMPTS[language].format(note=note)
    facts = judge.ask(prompt, _FACTS_SCHEMA)["facts"]
    return [fact.strip() for fact in facts if fact.strip()]


def _count_supported(
    judge: Asker, facts: Sequence[str], transcript: str, language: str
) -> int:
    """Ask about each fact against the whole transcript, one call each."""
    supported = 0
    for fact in facts:
        prompt = _VERDICT_PROMPTS[language].format(transcript=transcript, fact=fact)
        supported += judge.ask(prompt, _VERDICT_SCHEMA)["supported"] == "YES"
    return supported


@dataclass(frozen=True)
class Grounding:
    """Share of a note's facts the reference transcript supports. Needs a judge."""

    name: str = "grounding"
    group: str = "note_grounding"
    version: int = 1
    requires: Requires = Requires(("note",), ("transcript",))
    judge: Asker | None = None

    def with_judge(self, judge: Asker) -> "Grounding":
        """Return a copy that can score."""
        return replace(self, judge=judge)

    def describe(self, language: str) -> dict[str, object]:
        """Report the prompt language, checking one exists before any model loads.

        Raises:
            ValueError: If there are no prompts for the dataset's language.
        """
        if language not in _EXTRACT_PROMPTS:
            available = ", ".join(sorted(_EXTRACT_PROMPTS))
            raise ValueError(
                f"Metric {self.name!r} has prompts for {available}, not {language!r}."
            )
        return {"prompt_language": language, "validated": False}

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Check the note's facts; a note with no facts gets no score."""
        language = sample.language
        facts = _extract(self.judge, prediction.artifacts["note"].text, language)
        if not facts:
            return None
        transcript = _transcript_text(sample)
        supported = _count_supported(self.judge, facts, transcript, language)
        details = {"supported": float(supported), "facts": float(len(facts))}
        return Score(self.name, supported / len(facts), details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average per note, and pool facts across notes.

        The mean weighs every note equally; the pooled share weighs every fact, so
        long notes count more.
        """
        supported = sum(score.details["supported"] for score in scores)
        facts = sum(score.details["facts"] for score in scores)
        return {
            "mean": sum(score.value for score in scores) / len(scores),
            "pooled": supported / facts,
        }
