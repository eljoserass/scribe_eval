"""Checks on a note's shape, which need no reference."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

import textstat

from ..types import Prediction, Requires, Sample, Score
from .text import mean_values

_SECTIONS = (
    ("S", "Subjective", "Subjetivo"),
    ("O", "Objective", "Objetivo"),
    ("A", "Assessment", "Analysis", "Análisis"),
    ("P", "Plan"),
)


def _section_pattern(names: tuple[str, ...]) -> re.Pattern:
    """Match a section label at the start of a line, with optional markdown."""
    return re.compile(
        rf"^[\s*#_]*(?:{'|'.join(names)})\b\s*[*_]*\s*[—:\-–]", re.MULTILINE
    )


_PATTERNS = tuple(_section_pattern(names) for names in _SECTIONS)


def section_positions(text: str) -> list[int] | None:
    """Find S, O, A, and P in order; None if any is missing or out of order."""
    positions = []
    for pattern in _PATTERNS:
        match = pattern.search(text)
        if match is None:
            return None
        positions.append(match.start())
    return positions if positions == sorted(positions) else None


@dataclass(frozen=True)
class SoapSections:
    """Whether a note has the four sections in order, with nothing before them."""

    name: str = "soap_sections"
    group: str = "note_format"
    version: int = 1
    requires: Requires = Requires(("note",))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Give 1 to a well-formed note, 0 otherwise."""
        text = prediction.artifacts["note"].text.strip()
        positions = section_positions(text)
        return Score(self.name, float(positions is not None and positions[0] == 0))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the share of well-formed notes."""
        return {"rate": fmean(score.value for score in scores)}


@dataclass(frozen=True)
class NoteWords:
    """Note length in words, split on whitespace."""

    name: str = "note_words"
    group: str = "note_format"
    version: int = 1
    requires: Requires = Requires(("note",))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Count words."""
        return Score(self.name, float(len(prediction.artifacts["note"].text.split())))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report mean and longest note."""
        return {**mean_values(scores), "max": max(score.value for score in scores)}


@dataclass(frozen=True)
class FleschReadingEase:
    """Readability from sentence and word length. Higher reads easier.

    Clinical shorthand breaks its assumptions, so compare it between notes rather than
    reading it as an absolute.
    """

    name: str = "flesch_reading_ease"
    group: str = "note_format"
    version: int = 1
    requires: Requires = Requires(("note",))

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Score a note with the formula for its language."""
        textstat.set_lang(sample.language)
        value = textstat.flesch_reading_ease(prediction.artifacts["note"].text)
        return Score(self.name, float(value))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average scores."""
        return mean_values(scores)


@dataclass(frozen=True)
class CompressionRatio:
    """Note words per word of the reference conversation. Lower is more compact."""

    name: str = "compression_ratio"
    group: str = "note_format"
    version: int = 1
    requires: Requires = Requires(("note",), ("transcript",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Compare lengths in words."""
        note = len(prediction.artifacts["note"].text.split())
        conversation = len(sample.artifact("transcript").text.split())
        if not conversation:
            return None
        details = {"note": float(note), "conversation": float(conversation)}
        return Score(self.name, note / conversation, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Divide total note words by total conversation words."""
        note = sum(score.details["note"] for score in scores)
        conversation = sum(score.details["conversation"] for score in scores)
        return {"global": note / conversation}
