"""Checks on a note's shape, which need no reference.

`flesch_reading_ease` is Flesch, "A new readability yardstick", Journal of Applied
Psychology 32(3), 1948, through `textstat`, which adapts it per language.
`compression_ratio` stands in for brevity in Wang et al.'s framework, which doesn't
define it. `note_sections` is this project's check.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import fmean

import textstat

from ..types import Prediction, Requires, Sample, Score
from .models import apply_settings
from .text import mean_values

# SOAP in English: each section lists its accepted headings. Other languages and
# templates set their own headings in [metric_settings.note_sections].
SOAP_SECTIONS: tuple[tuple[str, ...], ...] = (
    ("S", "Subjective"),
    ("O", "Objective"),
    ("A", "Assessment"),
    ("P", "Plan"),
)


@lru_cache(maxsize=64)
def _heading_pattern(headings: tuple[str, ...]) -> re.Pattern:
    """Match a heading at the start of a line, with optional Markdown.

    A colon or dash may follow, or nothing else on the line: `S:`, `**Plan** -`, and
    `## Subjective` all match. Case is ignored except for one-letter headings, so a
    line reading `a` isn't taken for `A`.
    """
    names = "|".join(_heading_regex(heading) for heading in headings)
    return re.compile(
        rf"^[ \t*#_]*(?:{names})\b[ \t*_]*(?:[—:\-–]|$)",
        re.MULTILINE,
    )


def _heading_regex(heading: str) -> str:
    """Escape a heading, ignoring case unless it is a single letter."""
    escaped = re.escape(heading)
    return escaped if len(heading) == 1 else f"(?i:{escaped})"


def section_positions(
    text: str, sections: tuple[tuple[str, ...], ...] = SOAP_SECTIONS
) -> list[int | None]:
    """Find where each section's first heading starts; None for a missing one."""
    positions = []
    for headings in sections:
        match = _heading_pattern(headings).search(text)
        positions.append(None if match is None else match.start())
    return positions


def _sections_setting(value: object) -> tuple[tuple[str, ...], ...]:
    """Check `sections` from TOML: a list of sections, each a list of headings.

    Raises:
        ValueError: If it is empty, or a section has no headings or a non-string one.
    """
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("sections must be a non-empty list of heading lists.")
    sections = []
    for headings in value:
        if isinstance(headings, str):
            headings = [headings]
        valid = all(isinstance(heading, str) and heading for heading in headings)
        if not headings or not valid:
            raise ValueError("Each section needs one or more non-empty headings.")
        sections.append(tuple(headings))
    return tuple(sections)


@dataclass(frozen=True)
class NoteSections:
    """Whether a note has the expected sections in order, with nothing before them.

    The structure is a setting, so any template works; the default is SOAP. A section
    counts when one of its headings starts a line. It can't see junk after the last
    section.
    """

    name: str = "note_sections"
    group: str = "note_format"
    version: int = 1
    requires: Requires = Requires(("note",))

    # Settings, from [metric_settings.note_sections].
    sections: tuple[tuple[str, ...], ...] = SOAP_SECTIONS

    def with_settings(self, settings: Mapping[str, object]) -> "NoteSections":
        """Return a copy with another structure.

        Raises:
            ValueError: If a setting is unknown or `sections` is malformed.
        """
        checked = dict(settings)
        if "sections" in checked:
            checked["sections"] = _sections_setting(checked["sections"])
        return apply_settings(self, checked, ("sections",))

    def describe(self, language: str) -> dict[str, object]:
        """Report the expected sections, for the manifest."""
        return {"sections": [list(headings) for headings in self.sections]}

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Give 1 to a well-formed note, 0 otherwise, counting sections found."""
        text = prediction.artifacts["note"].text.strip()
        positions = section_positions(text, self.sections)
        found = [position for position in positions if position is not None]
        complete = len(found) == len(positions)
        well_formed = complete and found == sorted(found) and found[0] == 0
        details = {"found": float(len(found)), "expected": float(len(positions))}
        return Score(self.name, float(well_formed), details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the share of well-formed notes and of expected sections found."""
        found = sum(score.details["found"] for score in scores)
        expected = sum(score.details["expected"] for score in scores)
        return {
            "rate": fmean(score.value for score in scores),
            "coverage": found / expected,
        }


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
