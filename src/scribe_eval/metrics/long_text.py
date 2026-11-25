"""Split long notes into pieces a model can read, and combine the pieces' scores.

BERTScore, BLEURT, and NegBLEURT read at most 512 tokens. Each metric's `long_text`
setting picks one of these strategies, and accepts only those that fit how the
model was trained:

- `truncate`: score what fits and count the cut pairs. The default.
- `window`: embed overlapping token windows and match tokens over the whole text.
  MedBERTScore-SP: Ben Abacha et al., "An Investigation of Evaluation Methods in
  Automatic Medical Note Generation", Findings of ACL 2023.
  https://aclanthology.org/2023.findings-acl.161/
- `sentences`: score every reference sentence against every generated sentence and
  combine them as SMART-1 does: Amplayo et al., "SMART: Sentences as Basic Units for
  Text Evaluation", ICLR 2023. https://arxiv.org/abs/2208.01030
- `chunks`: score each generated sentence against reference chunks of about 350
  words, keep its best chunk, and average, as AlignScore does: Zha et al.,
  "AlignScore: Evaluating Factual Consistency with A Unified Alignment Function",
  ACL 2023. https://arxiv.org/abs/2305.16739

Per-section scoring, as ACI-BENCH and MEDIQA-Chat do, would be another strategy once
notes can be split into sections. Nothing here loads a model; `semantic.py` does.
Sentence splitting uses NLTK's Punkt models, which download on first use.
"""

from collections.abc import Sequence
from functools import lru_cache
from statistics import fmean

import nltk
from nltk.tokenize import sent_tokenize

TRUNCATE = "truncate"
WINDOW = "window"
SENTENCES = "sentences"
CHUNKS = "chunks"

# MedBERTScore-SP's overlap between consecutive windows, in tokens.
WINDOW_OVERLAP_TOKENS = 100

# AlignScore's chunk size, in words split on whitespace, as its code counts them.
CHUNK_WORDS = 350

# The Punkt models in NLTK's punkt_tab, by the ISO 639-1 code datasets record.
PUNKT_LANGUAGES = {
    "cs": "czech",
    "da": "danish",
    "de": "german",
    "el": "greek",
    "en": "english",
    "es": "spanish",
    "et": "estonian",
    "fi": "finnish",
    "fr": "french",
    "it": "italian",
    "ml": "malayalam",
    "nl": "dutch",
    "no": "norwegian",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "sl": "slovene",
    "sv": "swedish",
    "tr": "turkish",
}


def check_strategy(metric: str, strategy: str, accepted: tuple[str, ...]) -> None:
    """Reject a `long_text` strategy that a metric doesn't take.

    Raises:
        ValueError: If `strategy` isn't in `accepted`.
    """
    if strategy not in accepted:
        choices = ", ".join(repr(name) for name in accepted)
        raise ValueError(
            f"Metric {metric!r} takes long_text = {choices}, not {strategy!r}."
        )


def punkt_language(language: str, metric: str) -> str:
    """Name the Punkt model for a dataset language.

    Args:
        language: ISO 639-1 code, as in `dataset.json`.
        metric: The metric that splits sentences, for the error message.

    Raises:
        ValueError: If NLTK has no Punkt model for the language.
    """
    if language not in PUNKT_LANGUAGES:
        raise ValueError(
            f"Metric {metric!r} splits sentences with NLTK's Punkt, which has no "
            f"model for {language!r}. Use long_text = 'truncate' for this dataset."
        )
    return PUNKT_LANGUAGES[language]


@lru_cache(maxsize=1)
def _ensure_punkt() -> None:
    """Download NLTK's Punkt models once, unless they are already there."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def split_sentences(text: str, language: str, *, metric: str) -> list[str]:
    """Split a note into sentences: at line breaks first, then with Punkt.

    Notes hold headings and lists without final punctuation, which Punkt alone
    would merge into one long sentence, so every non-empty line is its own unit.
    May download Punkt's models on first use.

    Args:
        text: The note.
        language: ISO 639-1 code of the dataset.
        metric: The metric asking, for the error message.

    Returns:
        Sentences in order, stripped; empty when the text is blank.

    Raises:
        ValueError: If NLTK has no Punkt model for the language.
    """
    name = punkt_language(language, metric)
    _ensure_punkt()
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.extend(sent_tokenize(line, language=name))
    return sentences


def reference_chunks(
    sentences: Sequence[str], *, chunk_words: int = CHUNK_WORDS
) -> list[str]:
    """Join consecutive sentences into chunks of about `chunk_words` words.

    AlignScore's rule: ask for `words // chunk_words + 1` chunks and put the same
    number of whole sentences in each, so chunk lengths vary, and leftover
    sentences make one more chunk.

    Args:
        sentences: The reference's sentences, in order.
        chunk_words: Target words per chunk.

    Returns:
        Chunks in order; one chunk when the text is short.
    """
    words = sum(len(sentence.split()) for sentence in sentences)
    wanted = words // chunk_words + 1
    per_chunk = max(len(sentences) // wanted, 1)
    return [
        " ".join(sentences[start : start + per_chunk])
        for start in range(0, len(sentences), per_chunk)
    ]


def best_match_mean(scores: Sequence[Sequence[float]]) -> float:
    """Average each generated piece's best score against any reference piece.

    This is AlignScore's aggregation, and SMART-1's precision.

    Args:
        scores: `scores[i][j]` for reference piece `i` and generated sentence `j`.
    """
    return fmean(max(column) for column in zip(*scores, strict=True))


def smart_1(scores: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Combine sentence-pair scores into precision, recall, and F1, as SMART-1.

    Args:
        scores: `scores[i][j]` for reference sentence `i` and generated sentence
            `j`. SMART uses one matrix for both directions when the matching
            function is treated as symmetric, as its BLEURT experiments do.

    Returns:
        Precision, each generated sentence's best match averaged; recall, each
        reference sentence's best match averaged; and their harmonic mean, 0 when
        both add up to 0.
    """
    precision = best_match_mean(scores)
    recall = fmean(max(row) for row in scores)
    total = precision + recall
    f1 = 2 * precision * recall / total if total else 0.0
    return precision, recall, f1


def token_windows(
    length: int, *, size: int, overlap: int = WINDOW_OVERLAP_TOKENS
) -> list[tuple[int, int, int]]:
    """Plan overlapping windows over a sequence of tokens.

    Each window holds at most `size` tokens and starts with the last `overlap`
    tokens of the one before, as context. A token takes its embedding from the
    first window that holds it.

    Args:
        length: Tokens in the text, special tokens excluded.
        size: Tokens per window, special tokens excluded.
        overlap: Tokens each window repeats from the one before.

    Returns:
        `(start, first_new, end)` per window: it holds tokens `[start, end)`, and
        tokens `[first_new, end)` take their embeddings from it. One window when
        the text fits.

    Raises:
        ValueError: If `size` isn't larger than `overlap`.
    """
    if size <= overlap:
        raise ValueError(
            f"Windows of {size} tokens can't overlap by {overlap}; "
            "the model's limit is too small for long_text = 'window'."
        )
    windows = [(0, 0, min(size, length))]
    while windows[-1][2] < length:
        first_new = windows[-1][2]
        start = first_new - overlap
        windows.append((start, first_new, min(start + size, length)))
    return windows
