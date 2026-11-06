"""Transcription error rates against a reference transcript."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from statistics import fmean

import jiwer

from ..types import Prediction, Requires, Sample, Score, find_artifact
from .text import normalize_text


@dataclass(frozen=True)
class ErrorRate:
    """Edit distance over words or characters, after normalizing both texts."""

    name: str
    align: Callable
    group: str = "transcription"
    version: int = 1
    requires: Requires = Requires(("transcript",), ("transcript",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Align one transcript with its reference; skip an empty reference."""
        reference = normalize_text(sample.artifact("transcript").text, sample.language)
        predicted = find_artifact(prediction.artifacts, "transcript")
        hypothesis = normalize_text(predicted.text, sample.language)
        if not reference:
            return None
        if not hypothesis:
            size = float(self.align(reference, reference).hits)
            details = {"errors": size, "reference_size": size}
            return Score(self.name, 1.0, details)
        output = self.align(reference, hypothesis)
        errors = output.substitutions + output.deletions + output.insertions
        size = output.hits + output.substitutions + output.deletions
        details = {
            "errors": float(errors),
            "reference_size": float(size),
            "substitutions": float(output.substitutions),
            "deletions": float(output.deletions),
            "insertions": float(output.insertions),
        }
        return Score(self.name, errors / size, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report both views: long consultations weigh more in the global rate."""
        errors = sum(score.details["errors"] for score in scores)
        size = sum(score.details["reference_size"] for score in scores)
        return {
            "global": errors / size,
            "sample_mean": fmean(score.value for score in scores),
        }


@lru_cache(maxsize=4)
def _term_list(language: str) -> tuple[frozenset[str], int] | None:
    """Load a language's terms in normalized form, with the longest term's length."""
    path = files("scribe_eval") / "metrics" / "assets" / f"medical_terms_{language}.txt"
    if not path.is_file():
        return None
    lines = path.read_text(encoding="utf-8").splitlines()
    terms = {normalize_text(line, language) for line in lines} - {""}
    return frozenset(terms), max(len(term.split()) for term in terms)


def keep_terms(text: str, terms: frozenset[str], longest: int) -> list[str]:
    """Keep only medical terms, longest match first, each as one token."""
    words = text.split()
    kept = []
    position = 0
    while position < len(words):
        for size in range(min(longest, len(words) - position), 0, -1):
            candidate = " ".join(words[position : position + size])
            if candidate in terms:
                kept.append(candidate.replace(" ", "_"))
                position += size
                break
        else:
            position += 1
    return kept


@dataclass(frozen=True)
class MedicalTermErrorRate:
    """WER over medical terms only, following MedWER's protocol.

    Both texts keep only terms from a fixed list, then get aligned. A missed phrase
    counts as one error, and an invented term as an insertion. Terms outside the list
    are invisible, and languages without a list don't apply. Normalization differs from
    MedWER's, so the numbers aren't comparable with theirs.
    """

    name: str = "medical_term_wer"
    group: str = "transcription"
    version: int = 1
    requires: Requires = Requires(("transcript",), ("transcript",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Align the terms of one transcript with its reference."""
        term_list = _term_list(sample.language)
        if term_list is None:
            return None
        terms, longest = term_list
        reference_text = sample.artifact("transcript").text
        predicted_text = find_artifact(prediction.artifacts, "transcript").text
        reference = keep_terms(
            normalize_text(reference_text, sample.language), terms, longest
        )
        predicted = keep_terms(
            normalize_text(predicted_text, sample.language), terms, longest
        )
        if not reference and not predicted:
            return None
        if not reference:
            details = {"errors": float(len(predicted)), "reference_size": 0.0}
            return Score(self.name, 0.0, details)
        if not predicted:
            size = float(len(reference))
            return Score(self.name, 1.0, {"errors": size, "reference_size": size})
        output = jiwer.process_words(" ".join(reference), " ".join(predicted))
        errors = output.substitutions + output.deletions + output.insertions
        details = {"errors": float(errors), "reference_size": float(len(reference))}
        return Score(self.name, errors / len(reference), details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Add up errors across samples, as MedWER does."""
        errors = sum(score.details["errors"] for score in scores)
        size = sum(score.details["reference_size"] for score in scores)
        return {"global": errors / size if size else 0.0, "terms": size}


WORD_ERROR_RATE = ErrorRate("wer", jiwer.process_words)
CHARACTER_ERROR_RATE = ErrorRate("cer", jiwer.process_characters)
