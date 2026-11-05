"""Lexical comparison with a reference note. A diagnostic, not a safety signal."""

from collections.abc import Sequence
from dataclasses import dataclass

from ..types import Prediction, Requires, Sample, Score
from .text import mean_values, normalize_text


def _longest_common_subsequence(first: list[str], second: list[str]) -> int:
    """Measure the longest shared word sequence."""
    previous = [0] * (len(second) + 1)
    for word in first:
        current = [0]
        for index, other in enumerate(second, start=1):
            if word == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class RougeL:
    """ROUGE-L F1, on normalized words."""

    name: str = "rouge_l"
    group: str = "note_reference"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Compare a note with its reference."""
        generated = normalize_text(prediction.artifacts["note"].text, sample.language)
        reference = normalize_text(sample.artifact("note").text, sample.language)
        first, second = generated.split(), reference.split()
        if not first or not second:
            return Score(self.name, 0.0)
        common = _longest_common_subsequence(first, second)
        return Score(self.name, 2 * common / (len(first) + len(second)))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average scores."""
        return mean_values(scores)
