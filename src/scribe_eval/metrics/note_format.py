"""Checks on a note's shape, which need no reference."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

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
