"""Whisper's text normalizers, copied verbatim from openai/whisper.

Source: https://github.com/openai/whisper/tree/86098128c0b4f24f0e2aa2994de830614b474227/whisper/normalizers
License: MIT, Copyright (c) 2022 OpenAI; see LICENSE in this directory.

Radford et al., "Robust Speech Recognition via Large-Scale Weak Supervision",
ICML 2023. The Open ASR Leaderboard (Srivastav et al., arXiv:2510.06961) scores
English with `EnglishTextNormalizer` and other languages with
`BasicTextNormalizer(remove_diacritics=False)`. Don't edit these files; update them
from a newer commit instead, and bump the version of every metric that uses them.
"""

from .basic import BasicTextNormalizer as BasicTextNormalizer
from .english import EnglishTextNormalizer as EnglishTextNormalizer
