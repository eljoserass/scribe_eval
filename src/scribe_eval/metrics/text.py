"""Text normalization shared by metrics, and the default aggregation."""

import re
import unicodedata
from collections.abc import Sequence
from statistics import fmean

from num2words import num2words

from ..types import Score

_DIGITS = re.compile(r"\d+")
_FILLERS = {
    "en": {
        "ah",
        "eh",
        "er",
        "erm",
        "hm",
        "hmm",
        "mhm",
        "mm",
        "mmm",
        "uh",
        "uhm",
        "um",
        "umm",
    },
    "es": {"ah", "eh", "ehm", "hm", "hmm", "mhm", "mm", "mmm", "um"},
}


def _spell_number(match: re.Match, language: str) -> str:
    """Spell out a number, so `3` and `three` compare as equal."""
    return " " + num2words(int(match[0]), lang=language) + " "


def normalize_text(text: str, language: str) -> str:
    """Prepare text for comparison.

    Lowercases, spells out numbers, and drops punctuation and hesitations. Human
    transcripts keep hesitations and ASR usually doesn't, which isn't a real error.
    """
    text = unicodedata.normalize("NFC", text).casefold()
    text = _DIGITS.sub(lambda match: _spell_number(match, language), text)
    kept = []
    for character in text:
        if character == "'":
            kept.append(character)
        elif unicodedata.category(character)[0] in "PS":
            kept.append(" ")
        else:
            kept.append(character)
    fillers = _FILLERS.get(language, set())
    return " ".join(word for word in "".join(kept).split() if word not in fillers)


def mean_values(scores: Sequence[Score]) -> dict[str, float]:
    """Average scores."""
    return {"mean": fmean(score.value for score in scores)}
