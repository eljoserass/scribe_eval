"""Transcription error rates against a reference transcript."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
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


WORD_ERROR_RATE = ErrorRate("wer", jiwer.process_words)
CHARACTER_ERROR_RATE = ErrorRate("cer", jiwer.process_characters)
