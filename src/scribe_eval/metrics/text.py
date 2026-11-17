"""Text normalization shared by metrics, and the default aggregation.

Normalization follows the Open ASR Leaderboard (Srivastav et al., "Open ASR
Leaderboard: Towards Reproducible and Transparent Multilingual and Long-Form Speech
Recognition Evaluation", arXiv:2510.06961, 2025), which uses Whisper's normalizers
(Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision", ICML
2023):

- English: `EnglishTextNormalizer`. Lowercases, drops punctuation and hesitations,
  writes numbers as digits, and standardizes British spellings.
- Other languages: `BasicTextNormalizer`, keeping diacritics, after writing digits
  out as words with `num2words`, as the leaderboard does when it knows the language.
  Hesitations stay, since no standard list exists for other languages.
"""

import re
from collections.abc import Sequence
from functools import lru_cache
from statistics import fmean
from typing import Callable

from num2words import num2words

from ..types import Score
from .whisper_normalizers import BasicTextNormalizer, EnglishTextNormalizer

_DIGITS = re.compile(r"\d+")


def spell_numbers(text: str, language: str) -> str:
    """Write each run of digits as words, so `3` and `tres` compare as equal.

    Languages `num2words` doesn't know keep their digits.
    """

    def spell(match: re.Match) -> str:
        """Spell one number."""
        return " " + num2words(int(match[0]), lang=language) + " "

    try:
        return _DIGITS.sub(spell, text)
    except NotImplementedError:
        return text


@lru_cache(maxsize=8)
def _normalizer(language: str) -> Callable[[str], str]:
    """Build the normalizer for a language once."""
    if language == "en":
        return EnglishTextNormalizer()
    basic = BasicTextNormalizer(remove_diacritics=False)
    return lambda text: basic(spell_numbers(text, language))


def normalize_text(text: str, language: str) -> str:
    """Prepare text for comparison, the same way for prediction and reference.

    Args:
        text: Transcript or term to normalize.
        language: ISO 639-1 code of the sample, such as `en` or `es`.

    Returns:
        Lowercase words separated by single spaces, without punctuation.
    """
    return _normalizer(language)(text).strip()


def mean_values(scores: Sequence[Score]) -> dict[str, float]:
    """Average scores."""
    return {"mean": fmean(score.value for score in scores)}
