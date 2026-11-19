"""Clinical BERTScore for transcripts.

Shor et al., "Clinical BERTScore: An Improved Measure of Automatic Speech Recognition
Performance in Clinical Settings", ClinicalNLP 2023. https://arxiv.org/abs/2303.05737

    CBERTScore = k * BERTScore_medical + (1 - k) * BERTScore_all

`BERTScore_all` weighs every token; `BERTScore_medical` only tokens of medical terms
and numbers. Without either in a pair, k is 0. The paper tuned k = 0.4 on its
clinician preferences.

Differences from the paper, which released no code:

- Its 20K medical words come from Google's Knowledge Graph and aren't public. This
  uses MedWER's list, or the file in `terms`. English only by default.
- It scored single sentences. A consultation is too long for BERT, so the transcript
  is split along reference utterances, following a word alignment, and each pair is
  scored. The sample's value weighs utterances by reference words.
- Texts are normalized first, as for WER. Numbers count as medical whether
  normalization leaves them as digits, in English, or spells them out.

Medical tokens are weighted through BERTScore's IDF weights: every occurrence of a
token from a medical word counts, including inside a non-medical word. If one side of
a pair has no medical tokens, the package scores its medical part 0.
"""

import io
from collections import defaultdict
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from statistics import fmean

import jiwer

from ..types import Prediction, Requires, Sample, Score, find_artifact
from .models import LoadedModels, apply_settings, require_semantic, resolve_device
from .text import normalize_text


def split_by_reference(
    reference_segments: Sequence[str], hypothesis: str
) -> list[tuple[str, str]]:
    """Pair each reference utterance with the hypothesis words aligned to it.

    Matched and substituted words follow their reference word. An inserted word
    joins the utterance of the reference word before it.

    Args:
        reference_segments: Normalized reference utterances, in order.
        hypothesis: The normalized predicted transcript.

    Returns:
        (reference, hypothesis) pairs for non-empty utterances. A hypothesis is
        empty when everything in that utterance was missed.
    """
    owners = []
    words = []
    for index, text in enumerate(reference_segments):
        segment_words = text.split()
        words.extend(segment_words)
        owners.extend([index] * len(segment_words))
    if not words:
        return []
    spans: list[list[str]] = [[] for _ in reference_segments]
    hypothesis_words = hypothesis.split()
    if hypothesis_words:
        output = jiwer.process_words(" ".join(words), hypothesis)
        for chunk in output.alignments[0]:
            for offset in range(chunk.hyp_end_idx - chunk.hyp_start_idx):
                if chunk.type == "insert":
                    anchor = max(chunk.ref_start_idx - 1, 0)
                else:
                    anchor = chunk.ref_start_idx + offset
                spans[owners[anchor]].append(
                    hypothesis_words[chunk.hyp_start_idx + offset]
                )
    return [
        (text, " ".join(span))
        for text, span in zip(reference_segments, spans, strict=True)
        if text.split()
    ]


@lru_cache(maxsize=4)
def _number_words(language: str) -> frozenset[str]:
    """Numbers that normalize to one word, like `tres` or `veintiuno`.

    English normalization writes numbers as digits, so this is empty there. Words that
    only appear inside longer spellings, like `y` in `treinta y uno`, are left out.
    """
    words = set()
    for number in (*range(0, 101), 1000, 1_000_000):
        spelled = normalize_text(str(number), language).split()
        if len(spelled) == 1 and not spelled[0].isdigit():
            words.add(spelled[0])
    return frozenset(words)


def _is_number(word: str, numbers: frozenset[str]) -> bool:
    """Tell whether a normalized word is a number, in digits or spelled out."""
    return any(character.isdigit() for character in word) or word in numbers


@lru_cache(maxsize=4)
def keep_terms(text: str, terms: frozenset[str], longest: int) -> list[str]:
    """Keep only medical terms, longest match first, each as one token."""
    words = text.split()
    kept = []
    position = 0
    while position < len(words):
        for size in range(min(longest, len(words) - position), 0, -1):
            candidate = " ".join(words[position : position + size])
            if candidate in terms:
                kept.append(candidate.replace(" ", "_"))
                position += size
                break
        else:
            position += 1
    return kept


def _load_terms(path: str, language: str) -> tuple[frozenset[str], int]:
    """Read a term list in normalized form, with the longest term's length."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    terms = {normalize_text(line, language) for line in lines} - {""}
    if not terms:
        raise ValueError(f"No terms in {path}.")
    return frozenset(terms), max(len(term.split()) for term in terms)


def medical_words(
    text: str, terms: frozenset[str], longest: int, language: str
) -> set[str]:
    """Words of a normalized text that belong to a medical term or spell a number."""
    found = {
        word for term in keep_terms(text, terms, longest) for word in term.split("_")
    }
    numbers = _number_words(language)
    return found | {word for word in text.split() if _is_number(word, numbers)}


@dataclass(frozen=True)
class ClinicalBertScore:
    """BERTScore on transcripts with extra weight on medical terms and numbers."""

    name: str = "clinical_bertscore"
    group: str = "transcription_semantic"
    version: int = 2
    requires: Requires = Requires(("transcript",), ("transcript",))

    # Settings, from [metric_settings.clinical_bertscore].
    model: str | None = None
    num_layers: int | None = None
    medical_weight: float = 0.4
    terms: str | None = None
    device: str = "auto"
    batch_size: int = 64

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "ClinicalBertScore":
        """Copy with settings from the experiment; `terms` is a path, one per line."""
        allowed = (
            "model",
            "num_layers",
            "medical_weight",
            "terms",
            "device",
            "batch_size",
        )
        metric = apply_settings(self, settings, allowed)
        if not 0 <= metric.medical_weight <= 1:
            raise ValueError("medical_weight must be between 0 and 1.")
        return metric

    def _terms_path(self, language: str) -> str | None:
        """The configured list, or the packaged one for the language if any."""
        if self.terms is not None:
            return str(Path(self.terms).expanduser())
        path = (
            files("scribe_eval")
            / "metrics"
            / "assets"
            / f"medical_terms_{language}.txt"
        )
        return str(path) if path.is_file() else None

    def describe(self, language: str) -> dict[str, object]:
        """Model, layer, weight, and which term list, by hash."""
        require_semantic("bert_score", self.name)
        from bert_score.utils import lang2model, model2layers

        model = self.model or lang2model[language]
        path = self._terms_path(language)
        digest = sha256(Path(path).read_bytes()).hexdigest() if path else None
        return {
            "model": model,
            "num_layers": self.num_layers or model2layers.get(model),
            "medical_weight": self.medical_weight,
            "terms_sha256": digest,
            "bert_score": version("bert-score"),
        }

    def _encoder(self, language: str):
        """Load model and tokenizer once; downloads weights on first use."""

        def load():
            """Use the package's loaders, so embeddings match standard BERTScore."""
            from bert_score.utils import (
                get_model,
                get_tokenizer,
                lang2model,
                model2layers,
            )

            model_type = self.model or lang2model[language]
            layers = self.num_layers or model2layers[model_type]
            device = resolve_device(self.device)
            model = get_model(model_type, layers).to(device)
            return model, get_tokenizer(model_type), device

        return self._models.get(language, load)

    def _f1(
        self, encoder, pairs: list[tuple[str, str]], words: set[str] | None
    ) -> list[float]:
        """BERTScore F1 per pair, over all tokens or only tokens of `words`."""
        from bert_score.utils import bert_cos_score_idf, sent_encode

        model, tokenizer, device = encoder
        if words is None:
            weights = defaultdict(lambda: 1.0)
        else:
            weights = defaultdict(float)
            for word in words:
                for token in sent_encode(tokenizer, word)[1:-1]:
                    weights[token] = 1.0
        weights[tokenizer.sep_token_id] = 0.0
        weights[tokenizer.cls_token_id] = 0.0
        references = [reference for reference, _ in pairs]
        candidates = [candidate for _, candidate in pairs]
        # The package warns on stderr for every side left with no weighted token.
        with redirect_stderr(io.StringIO()):
            scores = bert_cos_score_idf(
                model,
                references,
                candidates,
                tokenizer,
                weights,
                batch_size=self.batch_size,
                device=device,
            )
        return [float(value) for value in scores[:, 2]]

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one transcript, utterance by utterance."""
        path = self._terms_path(sample.language)
        if path is None:
            return None
        terms, longest = _load_terms(path, sample.language)
        segments = [
            normalize_text(segment.text, sample.language)
            for segment in sample.artifact("transcript").segments
        ]
        predicted = find_artifact(prediction.artifacts, "transcript")
        pairs = split_by_reference(
            segments, normalize_text(predicted.text, sample.language)
        )
        if not pairs:
            return None

        medical = [
            medical_words(reference, terms, longest, sample.language)
            | medical_words(candidate, terms, longest, sample.language)
            for reference, candidate in pairs
        ]
        encoder = self._encoder(sample.language)
        overall = self._f1(encoder, pairs, None)
        any_medical = set().union(*medical)
        weighted = self._f1(encoder, pairs, any_medical) if any_medical else overall

        values = []
        for all_tokens, medical_tokens, found in zip(
            overall, weighted, medical, strict=True
        ):
            weight = self.medical_weight if found else 0.0
            values.append(weight * medical_tokens + (1 - weight) * all_tokens)
        sizes = [len(reference.split()) for reference, _ in pairs]
        total = sum(sizes)
        unweighted = sum(v * s for v, s in zip(overall, sizes, strict=True)) / total
        details = {
            "reference_words": float(total),
            "utterances": float(len(pairs)),
            "medical_utterances": float(sum(bool(found) for found in medical)),
            "bertscore": unweighted,
        }
        value = sum(v * s for v, s in zip(values, sizes, strict=True)) / total
        return Score(self.name, value, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Global weighs consultations by reference words; sample mean doesn't."""
        words = sum(score.details["reference_words"] for score in scores)
        return {
            "global": sum(s.value * s.details["reference_words"] for s in scores)
            / words,
            "sample_mean": fmean(score.value for score in scores),
            "bertscore_global": sum(
                s.details["bertscore"] * s.details["reference_words"] for s in scores
            )
            / words,
        }
