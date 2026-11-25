"""Repetition within one text, which needs no reference.

A generator stuck in a loop repeats itself until its token limit. These measures
count that directly:

- `seq_rep` and `transcript_seq_rep` are seq-rep-n from Welleck et al., "Neural Text
  Generation with Unlikelihood Training", ICLR 2020 (arXiv:1908.04319): one minus
  the share of a text's word n-grams that are unique.
- `repeated_lines` is a descriptive count, not a published metric: the share of a
  note's non-empty lines that appear in it more than once.

Repetition was the most frequent issue in the evaluators' feedback on generated
consultation notes in Moramarco et al., "Human Evaluation and Correlation with
Automatic Metrics in Consultation Note Generation", ACL 2022, Table 5.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field

from ..artifacts import ArtifactKind
from ..types import Prediction, Requires, Sample, Score, find_artifact
from .models import apply_settings
from .text import mean_values, normalize_text


def sequence_repetition(words: Sequence[str], *, n: int) -> tuple[float, int] | None:
    """Calculate seq-rep-n, one minus unique n-grams over all n-grams of one text.

    Args:
        words: The text's words, in order.
        n: N-gram length in words, at least one.

    Returns:
        The share, from 0 when no n-gram repeats toward 1, and the number of
        n-grams; None when the text has fewer than `n` words, so no n-grams.
    """
    ngrams = [tuple(words[start : start + n]) for start in range(len(words) - n + 1)]
    if not ngrams:
        return None
    return 1 - len(set(ngrams)) / len(ngrams), len(ngrams)


def repeated_line_share(text: str) -> tuple[float, int] | None:
    """Calculate the share of non-empty lines that appear more than once in a text.

    Lines compare after collapsing whitespace and lowercasing. Every occurrence of a
    repeated line counts, the first one included, so `A, B, A` gives 2/3.

    Args:
        text: The text, split on line breaks.

    Returns:
        The share, from 0 to 1, and the number of non-empty lines; None when the text
        has no non-empty line.
    """
    lines = [" ".join(line.split()).lower() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return None
    counts = Counter(lines)
    repeated = sum(1 for line in lines if counts[line] > 1)
    return repeated / len(lines), len(lines)


def _check_threshold(name: str, threshold: float | None) -> None:
    """Reject a threshold that isn't a share between 0 and 1.

    Raises:
        ValueError: If `threshold` is set and isn't a number from 0 to 1.
    """
    if threshold is None:
        return
    is_number = isinstance(threshold, int | float) and not isinstance(threshold, bool)
    if not is_number or not 0 <= threshold <= 1:
        raise ValueError(f"Metric {name!r}: threshold must be between 0 and 1.")


def _aggregate(scores: Sequence[Score], threshold: float | None) -> dict[str, float]:
    """Report the mean and the highest value, and how many exceed a set threshold.

    The maximum keeps a single runaway text visible beside the mean.
    """
    result = {**mean_values(scores), "max": max(score.value for score in scores)}
    if threshold is not None:
        result["above_threshold"] = sum(score.value > threshold for score in scores)
    return result


@dataclass(frozen=True)
class SequenceRepetition:
    """seq-rep-n of one artifact's text, over words after the metric normalization.

    The text is normalized as for WER, for the sample's language, then split on
    whitespace. An immutable configuration with no resources; `with_settings` makes
    the copy an experiment uses.
    """

    name: str
    kind: ArtifactKind
    group: str
    _: KW_ONLY
    version: int = 1
    requires: Requires = field(init=False)

    # Settings, from [metric_settings.<name>].
    n: int = 4
    threshold: float | None = None

    def __post_init__(self) -> None:
        """Derive what the metric needs: the artifact, and no reference."""
        object.__setattr__(self, "requires", Requires((self.kind,)))

    def with_settings(self, settings: Mapping[str, object]) -> "SequenceRepetition":
        """Copy with `n`, the n-gram length, and `threshold` from the experiment.

        Raises:
            ValueError: If a setting is unknown, `n` isn't a whole number of at least
                1, or `threshold` isn't between 0 and 1.
        """
        metric = apply_settings(self, settings, ("n", "threshold"))
        n = metric.n
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"Metric {self.name!r}: n must be a whole number >= 1.")
        _check_threshold(self.name, metric.threshold)
        return metric

    def describe(self, language: str) -> dict[str, object]:
        """Report the effective settings, for the manifest."""
        return {"n": self.n, "threshold": self.threshold}

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one text; one shorter than `n` words has no n-grams and no score."""
        text = find_artifact(prediction.artifacts, self.kind).text
        words = normalize_text(text, sample.language).split()
        result = sequence_repetition(words, n=self.n)
        if result is None:
            return None
        value, ngrams = result
        return Score(self.name, value, {"ngrams": float(ngrams)})

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the mean and max, and the count above `threshold` when set."""
        return _aggregate(scores, self.threshold)


@dataclass(frozen=True)
class RepeatedLines:
    """Share of a note's non-empty lines that appear more than once in it.

    Lines are compared as written, after collapsing whitespace and lowercasing, so
    numbering or punctuation that changes between copies hides a repeat. Notes only:
    in a transcript, line breaks follow each pipeline's segmentation, and short turns
    such as "okay" repeat in any conversation. An immutable configuration with no
    resources; `with_settings` makes the copy an experiment uses.
    """

    name: str = "repeated_lines"
    group: str = "note_format"
    version: int = 1
    requires: Requires = Requires(("note",))

    # Settings, from [metric_settings.repeated_lines].
    threshold: float | None = None

    def with_settings(self, settings: Mapping[str, object]) -> "RepeatedLines":
        """Copy with `threshold` from the experiment.

        Raises:
            ValueError: If a setting is unknown or `threshold` isn't between 0 and 1.
        """
        metric = apply_settings(self, settings, ("threshold",))
        _check_threshold(self.name, metric.threshold)
        return metric

    def describe(self, language: str) -> dict[str, object]:
        """Report the effective settings, for the manifest."""
        return {"threshold": self.threshold}

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one note; one without a non-empty line has no score."""
        result = repeated_line_share(prediction.artifacts["note"].text)
        if result is None:
            return None
        value, lines = result
        return Score(self.name, value, {"lines": float(lines)})

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the mean and max, and the count above `threshold` when set."""
        return _aggregate(scores, self.threshold)
