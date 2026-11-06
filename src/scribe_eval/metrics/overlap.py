"""Text overlap with a reference, through the standard libraries.

Scores come from `rouge-score`, `sacrebleu`, `nltk`, and `rapidfuzz` on raw text, so
they stay comparable with published numbers. These are diagnostics, not safety signals.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache

import nltk
from nltk.translate.meteor_score import meteor_score
from rapidfuzz.distance import Levenshtein as _Levenshtein
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

from ..artifacts import ArtifactKind
from ..types import Prediction, Requires, Sample, Score, find_artifact
from .text import mean_values

_NGRAMS = 4
_bleu = BLEU(effective_order=True)


class _WordTokenizer:
    """Keep accented letters, which rouge-score's English tokenizer drops."""

    def tokenize(self, text: str) -> list[str]:
        """Split lowercase text into words."""
        return re.findall(r"\w+", text.lower())


@lru_cache(maxsize=16)
def _rouge(rouge_type: str, language: str) -> rouge_scorer.RougeScorer:
    """Build a scorer, using the library's own tokenizer for English."""
    tokenizer = None if language == "en" else _WordTokenizer()
    return rouge_scorer.RougeScorer([rouge_type], tokenizer=tokenizer)


def _texts(
    prediction: Prediction, sample: Sample, kind: ArtifactKind
) -> tuple[str, str]:
    """Read the generated and reference text of an artifact kind."""
    generated = find_artifact(prediction.artifacts, kind).text
    return generated, sample.artifact(kind).text


def _requires(metric: "Rouge | Bleu") -> Requires:
    """Compare an artifact with its own reference."""
    return Requires((metric.kind,), (metric.kind,))


@dataclass(frozen=True)
class Rouge:
    """ROUGE F1."""

    name: str
    rouge_type: str
    kind: ArtifactKind
    group: str
    version: int = 1
    requires: Requires = field(init=False)

    def __post_init__(self) -> None:
        """Derive what the metric needs."""
        object.__setattr__(self, "requires", _requires(self))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Compare a text with its reference."""
        generated, reference = _texts(prediction, sample, self.kind)
        scorer = _rouge(self.rouge_type, sample.language)
        result = scorer.score(reference, generated)[self.rouge_type]
        return Score(self.name, result.fmeasure)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average scores."""
        return mean_values(scores)


@dataclass(frozen=True)
class Bleu:
    """BLEU, on sacrebleu's 0–100 scale."""

    name: str
    kind: ArtifactKind
    group: str
    version: int = 1
    requires: Requires = field(init=False)

    def __post_init__(self) -> None:
        """Derive what the metric needs."""
        object.__setattr__(self, "requires", _requires(self))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Score one text, keeping n-gram counts for corpus BLEU."""
        generated, reference = _texts(prediction, sample, self.kind)
        result = _bleu.sentence_score(generated, [reference])
        details = {"system_length": float(result.sys_len)}
        details["reference_length"] = float(result.ref_len)
        for order in range(_NGRAMS):
            details[f"correct_{order}"] = float(result.counts[order])
            details[f"total_{order}"] = float(result.totals[order])
        return Score(self.name, result.score, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report corpus BLEU, the standard form, and the mean of sample scores."""

        def total(key: str) -> int:
            """Add up one count."""
            return int(sum(score.details[key] for score in scores))

        corpus = BLEU.compute_bleu(
            correct=[total(f"correct_{order}") for order in range(_NGRAMS)],
            total=[total(f"total_{order}") for order in range(_NGRAMS)],
            sys_len=total("system_length"),
            ref_len=total("reference_length"),
            smooth_method="exp",
            effective_order=True,
        )
        return {"corpus": corpus.score, **mean_values(scores)}


@lru_cache(maxsize=1)
def _load_wordnet() -> None:
    """Fetch WordNet, which METEOR needs, on first use."""
    nltk.download("wordnet", quiet=True)


@dataclass(frozen=True)
class Meteor:
    """METEOR. Its synonyms come from WordNet, so it's English only."""

    name: str = "meteor"
    group: str = "note_reference"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Compare a note with its reference."""
        if sample.language != "en":
            return None
        _load_wordnet()
        generated, reference = _texts(prediction, sample, "note")
        words = _WordTokenizer()
        value = meteor_score([words.tokenize(reference)], words.tokenize(generated))
        return Score(self.name, value)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average scores."""
        return mean_values(scores)


@dataclass(frozen=True)
class Levenshtein:
    """Character edits between a note and its reference. Lower is better.

    The best simple metric against clinicians in PriMock57's human evaluation.
    """

    name: str = "levenshtein"
    group: str = "note_reference"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Count edits, keeping the size so a rate can be derived."""
        generated, reference = _texts(prediction, sample, "note")
        distance = float(_Levenshtein.distance(generated, reference))
        size = float(max(len(generated), len(reference), 1))
        return Score(self.name, distance, {"size": size})

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report mean edits, and edits per character."""
        edits = sum(score.value for score in scores)
        size = sum(score.details["size"] for score in scores)
        return {**mean_values(scores), "per_character": edits / size}
