"""Note similarity from neural models: BERTScore, BLEURT, and NegBLEURT.

Zhang et al., "BERTScore: Evaluating Text Generation with BERT", ICLR 2020.
https://arxiv.org/abs/1904.09675
Sellam et al., "BLEURT: Learning Robust Metrics for Text Generation", ACL 2020.
https://arxiv.org/abs/2004.04696
Anschütz, Miguel Lozano, and Groh, "This is not correct! Negation-aware Evaluation
of Language Generation Systems", INLG 2023.
https://aclanthology.org/2023.inlg-main.12/

All need the `semantic` extra and download their weights on first use. The default
models read 512 tokens. Past that, the `long_text` setting decides: `truncate`
scores what fits and flags it in `truncated`; BERTScore's `window`, BLEURT's
`sentences` and `chunks`, and NegBLEURT's `sentences` score whole notes. See
`long_text.py`.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import version
from statistics import fmean
from typing import TYPE_CHECKING

from ..types import Prediction, Requires, Sample, Score
from .long_text import (
    CHUNK_WORDS,
    CHUNKS,
    SENTENCES,
    TRUNCATE,
    WINDOW,
    WINDOW_OVERLAP_TOKENS,
    best_match_mean,
    check_strategy,
    punkt_language,
    reference_chunks,
    smart_1,
    split_sentences,
    token_windows,
)
from .models import LoadedModels, apply_settings, require_semantic, resolve_device

if TYPE_CHECKING:
    import torch
    from bert_score import BERTScorer
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Strategies each model takes. Token embeddings take windows, and pair models take
# sentences. NegBLEURT, trained on sentence pairs, scored chunk pairs below 0 even
# for a faithful note, so only BLEURT takes chunks.
_EMBEDDING_STRATEGIES = (TRUNCATE, WINDOW)
_PAIR_STRATEGIES = (TRUNCATE, SENTENCES, CHUNKS)
_SENTENCE_PAIR_STRATEGIES = (TRUNCATE, SENTENCES)

# Scores (references, candidates) pairs; returns one score per pair, and how many
# pairs were truncated.
_PairScorer = Callable[[Sequence[str], Sequence[str]], tuple[list[float], int]]


def _texts(prediction: Prediction, sample: Sample) -> tuple[str, str] | None:
    """Return the generated and reference notes, or None if either is empty."""
    candidate = prediction.artifacts["note"].text.strip()
    reference = sample.artifact("note").text.strip()
    return (candidate, reference) if candidate and reference else None


def _mean_details(scores: Sequence[Score], keys: tuple[str, ...]) -> dict[str, float]:
    """Average the value and some details, and count truncated inputs."""
    result = {"mean": fmean(score.value for score in scores)}
    for key in keys:
        result[key] = fmean(score.details[key] for score in scores)
    result["truncated"] = sum(score.details["truncated"] for score in scores)
    return result


def _bertscore_ids(tokenizer: "PreTrainedTokenizerBase", text: str) -> list[int]:
    """Encode a text as bert_score's `sent_encode` does, without special tokens.

    The package adds a prefix space for GPT-2 and RoBERTa tokenizers, and so does
    this, but it never truncates.
    """
    from transformers import GPT2Tokenizer, RobertaTokenizer

    options = {}
    if isinstance(tokenizer, (GPT2Tokenizer, RobertaTokenizer)):
        options["add_prefix_space"] = True
    return tokenizer.encode(text.strip(), add_special_tokens=False, **options)


def _window_embedding(
    content_ids: list[int],
    *,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    device: str,
    batch_size: int,
) -> tuple["torch.Tensor", list[int]]:
    """Embed a text window by window, as MedBERTScore-SP does.

    Each window is a complete input, special tokens included, of at most the
    tokenizer's `model_max_length` tokens. Consecutive windows share
    `WINDOW_OVERLAP_TOKENS` tokens, read as context only: each token takes its
    embedding from the first window that holds it. Special tokens come from the
    first window, before the text, and from the last, after it.

    Args:
        content_ids: The text's tokens, without special tokens; not empty.
        model: BERTScore's model, cut at its layer, in evaluation mode.
        tokenizer: The model's tokenizer.
        device: Where `model` is.
        batch_size: Windows per forward pass.

    Returns:
        One embedding per token of the whole input, and that input's token IDs. A
        text that fits gets the single input BERTScore gives it.
    """
    import torch
    from bert_score.utils import bert_encode, padding

    size = tokenizer.model_max_length - tokenizer.num_special_tokens_to_add()
    windows = token_windows(len(content_ids), size=size)
    inputs = [
        tokenizer.build_inputs_with_special_tokens(content_ids[start:end])
        for start, _, end in windows
    ]
    special = tokenizer.get_special_tokens_mask(content_ids[:1])
    prefix = special.index(0)

    embeddings = []
    for batch_start in range(0, len(inputs), batch_size):
        batch = inputs[batch_start : batch_start + batch_size]
        padded, _, mask = padding(batch, tokenizer.pad_token_id)
        embeddings.extend(
            bert_encode(model, padded.to(device), attention_mask=mask.to(device))
        )

    pieces = []
    ids = []
    for index, (start, first_new, end) in enumerate(windows):
        window_ids = inputs[index]
        first = 0 if index == 0 else prefix + first_new - start
        last = len(window_ids) if index == len(windows) - 1 else prefix + end - start
        pieces.append(embeddings[index][first:last])
        ids.extend(window_ids[first:last])
    return torch.cat(pieces), ids


def _windowed_bertscore(
    scorer: "BERTScorer",
    tokenizer: "PreTrainedTokenizerBase",
    *,
    candidate_ids: list[int],
    reference_ids: list[int],
    batch_size: int,
) -> tuple[float, float, float]:
    """BERTScore of two texts embedded window by window, as MedBERTScore-SP.

    Matching, token weights, and baseline rescaling are the package's, so only the
    embeddings differ from `scorer.score`.

    Args:
        scorer: The loaded scorer, whose model and baseline this uses.
        tokenizer: The model's tokenizer.
        candidate_ids: The generated note's tokens, without special tokens.
        reference_ids: The reference note's tokens, without special tokens.
        batch_size: Windows per forward pass.

    Returns:
        Precision, recall, and F1.
    """
    import torch
    from bert_score.utils import greedy_cos_idf

    # BERTScorer has no public accessor for its model; this avoids loading a copy.
    model = scorer._model
    # Without IDF, bert_score weighs every token 1 except these two.
    ignored = {tokenizer.cls_token_id, tokenizer.sep_token_id}
    stats = []
    for content_ids in (reference_ids, candidate_ids):
        embedding, ids = _window_embedding(
            content_ids,
            model=model,
            tokenizer=tokenizer,
            device=scorer.device,
            batch_size=batch_size,
        )
        weights = [0.0 if token in ignored else 1.0 for token in ids]
        mask = torch.ones(1, len(ids), dtype=torch.long, device=embedding.device)
        idf = torch.tensor([weights], device=embedding.device)
        stats.extend((embedding.unsqueeze(0), mask, idf))

    with torch.no_grad():
        values = torch.stack(greedy_cos_idf(*stats)).flatten().cpu()
    if scorer.rescale_with_baseline:
        values = (values - scorer.baseline_vals) / (1 - scorer.baseline_vals)
    precision, recall, f1 = (float(value) for value in values)
    return precision, recall, f1


def _regression_scores(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    device: str,
    references: Sequence[str],
    candidates: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
) -> tuple[list[float], int]:
    """Score pairs with a BLEURT-style model: BERT plus a regression head.

    Each pair is one input, reference first, as in BLEURT. Past `max_length` tokens,
    the longer text loses tokens first, as BLEURT's own encoder does.

    Args:
        model: Sequence classifier with a single output, in evaluation mode.
        tokenizer: The checkpoint's tokenizer.
        device: Where `model` is.
        references: The reference side of each pair.
        candidates: The generated side of each pair, in the same order.
        max_length: Token limit for a pair, special tokens included.
        batch_size: Pairs per forward pass.

    Returns:
        The model's output for each pair, in order, and how many were truncated.
    """
    import torch

    values = []
    truncated = 0
    for start in range(0, len(references), batch_size):
        batch = (
            list(references[start : start + batch_size]),
            list(candidates[start : start + batch_size]),
        )
        lengths = [len(ids) for ids in tokenizer(*batch)["input_ids"]]
        truncated += sum(length > max_length for length in lengths)
        inputs = tokenizer(
            *batch,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        values.extend(float(value) for value in logits.flatten())
    return values, truncated


def _regression_note_score(
    score_pairs: _PairScorer,
    *,
    reference: str,
    candidate: str,
    language: str,
    long_text: str,
    metric: str,
) -> tuple[float, dict[str, float]]:
    """Score a note against its reference with a pair model, by strategy.

    `truncate` scores the notes as one pair. `sentences` scores every reference
    sentence against every generated one and returns SMART-1's F1. `chunks` scores
    every generated sentence against every reference chunk and returns
    AlignScore's mean of best matches.

    Args:
        score_pairs: Scores (references, candidates) pairs with the model.
        reference: The reference note.
        candidate: The generated note.
        language: ISO 639-1 code of the dataset, for sentence splitting.
        long_text: The strategy.
        metric: The metric's name, for error messages.

    Returns:
        The note's value and its details: `truncated` always, 1 if any pair was
        cut; `pairs` when notes are split; `precision` and `recall` for sentences.
    """
    if long_text == TRUNCATE:
        values, truncated = score_pairs([reference], [candidate])
        return values[0], {"truncated": float(truncated > 0)}

    generated = split_sentences(candidate, language, metric=metric)
    reference_sentences = split_sentences(reference, language, metric=metric)
    if long_text == CHUNKS:
        pieces = reference_chunks(reference_sentences)
    else:
        pieces = reference_sentences
    references = []
    candidates = []
    for piece in pieces:
        for sentence in generated:
            references.append(piece)
            candidates.append(sentence)
    values, truncated = score_pairs(references, candidates)

    width = len(generated)
    matrix = [values[row : row + width] for row in range(0, len(values), width)]
    details = {"truncated": float(truncated > 0), "pairs": float(len(values))}
    if long_text == CHUNKS:
        return best_match_mean(matrix), details
    precision, recall, f1 = smart_1(matrix)
    return f1, {**details, "precision": precision, "recall": recall}


def _split_settings(long_text: str, language: str, metric: str) -> dict[str, object]:
    """What a strategy that splits notes adds to the manifest.

    Raises:
        ValueError: If the strategy splits sentences in a language Punkt lacks.
    """
    if long_text == TRUNCATE:
        return {}
    settings: dict[str, object] = {
        "punkt": punkt_language(language, metric),
        "nltk": version("nltk"),
    }
    if long_text == CHUNKS:
        settings["chunk_words"] = CHUNK_WORDS
    return settings


def _regression_aggregate(scores: Sequence[Score], long_text: str) -> dict[str, float]:
    """Average the scores, count notes with a truncated pair, and total the pairs."""
    keys = ("precision", "recall") if long_text == SENTENCES else ()
    result = _mean_details(scores, keys)
    if long_text != TRUNCATE:
        result["pairs"] = sum(score.details["pairs"] for score in scores)
    return result


@dataclass(frozen=True)
class BertScore:
    """F1 of token embeddings matched greedily by cosine similarity.

    Zhang et al., ICLR 2020, through their `bert-score` package. Without a `model`,
    it uses the package's default for the sample's language: `roberta-large` in
    English, multilingual BERT otherwise. `describe` gives what to report. Loaded
    models stay with the configured metric until scoring ends.

    `long_text = "window"` embeds texts past the model's limit in overlapping
    windows, as MedBERTScore-SP; pairs that fit score as with `truncate`.
    """

    name: str = "bertscore"
    group: str = "note_semantic"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))

    # Settings, from [metric_settings.bertscore].
    model: str | None = None
    num_layers: int | None = None
    rescale_with_baseline: bool = False
    device: str = "auto"
    batch_size: int = 64
    long_text: str = TRUNCATE

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "BertScore":
        """Copy with settings from the experiment.

        Raises:
            ValueError: If a setting is unknown or `long_text` isn't one it takes.
        """
        allowed = (
            "model",
            "num_layers",
            "rescale_with_baseline",
            "device",
            "batch_size",
            "long_text",
        )
        metric = apply_settings(self, settings, allowed)
        check_strategy(metric.name, metric.long_text, _EMBEDDING_STRATEGIES)
        return metric

    def describe(self, language: str) -> dict[str, object]:
        """Effective model, layer, and long-text strategy, for the manifest."""
        require_semantic("bert_score", self.name)
        from bert_score.utils import lang2model, model2layers

        model = self.model or lang2model[language]
        settings: dict[str, object] = {
            "model": model,
            "num_layers": self.num_layers or model2layers.get(model),
            "rescale_with_baseline": self.rescale_with_baseline,
            "long_text": self.long_text,
        }
        if self.long_text == WINDOW:
            settings["window_overlap_tokens"] = WINDOW_OVERLAP_TOKENS
        settings["bert_score"] = version("bert-score")
        return settings

    def _scorer(self, language: str):
        """Load the scorer for a language once; downloads weights on first use."""

        def load():
            """Build the package's scorer with this metric's settings."""
            from bert_score import BERTScorer
            from bert_score.utils import get_tokenizer

            scorer = BERTScorer(
                model_type=self.model,
                num_layers=self.num_layers,
                lang=language,
                rescale_with_baseline=self.rescale_with_baseline,
                device=resolve_device(self.device),
            )
            return scorer, get_tokenizer(scorer.model_type)

        return self._models.get(language, load)

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one note against its reference."""
        texts = _texts(prediction, sample)
        if texts is None:
            return None
        candidate, reference = texts
        scorer, tokenizer = self._scorer(sample.language)
        candidate_ids = _bertscore_ids(tokenizer, candidate)
        reference_ids = _bertscore_ids(tokenizer, reference)
        longest = max(len(candidate_ids), len(reference_ids))
        too_long = (
            longest + tokenizer.num_special_tokens_to_add() > tokenizer.model_max_length
        )

        if self.long_text == WINDOW and too_long:
            precision, recall, f1 = _windowed_bertscore(
                scorer,
                tokenizer,
                candidate_ids=candidate_ids,
                reference_ids=reference_ids,
                batch_size=self.batch_size,
            )
        else:
            values = scorer.score([candidate], [reference], batch_size=self.batch_size)
            precision, recall, f1 = (float(value[0]) for value in values)

        details = {
            "precision": precision,
            "recall": recall,
            "truncated": float(too_long and self.long_text == TRUNCATE),
        }
        if self.long_text == WINDOW:
            details["windowed"] = float(too_long)
        return Score(self.name, f1, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average F1, precision, and recall; count truncated or windowed pairs."""
        result = _mean_details(scores, ("precision", "recall"))
        if self.long_text == WINDOW:
            result["windowed"] = sum(score.details["windowed"] for score in scores)
        return result


@dataclass(frozen=True)
class Bleurt:
    """A learned score: BERT fine-tuned to predict human ratings of similarity.

    Sellam et al., ACL 2020. Google's code needs TensorFlow; this uses the
    `bleurt-pytorch` port and its conversions of the official checkpoints.
    BLEURT-20 is Google's recommended checkpoint; the D12, D6, and D3 distillations
    are smaller and faster. Scores roughly span 0 to 1, and aren't comparable across
    checkpoints. The loaded checkpoint stays with the configured metric until
    scoring ends.

    `long_text` takes `sentences` or `chunks` to score whole notes as sentence pairs.
    """

    name: str = "bleurt"
    group: str = "note_semantic"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))

    # Settings, from [metric_settings.bleurt].
    checkpoint: str = "lucadiliello/BLEURT-20"
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 16
    long_text: str = TRUNCATE

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "Bleurt":
        """Copy with settings from the experiment.

        Raises:
            ValueError: If a setting is unknown or `long_text` isn't one it takes.
        """
        allowed = ("checkpoint", "device", "max_length", "batch_size", "long_text")
        metric = apply_settings(self, settings, allowed)
        check_strategy(metric.name, metric.long_text, _PAIR_STRATEGIES)
        return metric

    def describe(self, language: str) -> dict[str, object]:
        """Checkpoint, long-text strategy, and package versions.

        Raises:
            ValueError: If the extra is missing, or the strategy splits sentences in
                a language Punkt lacks.
        """
        require_semantic("bleurt_pytorch", self.name)
        return {
            "checkpoint": self.checkpoint,
            "max_length": self.max_length,
            "long_text": self.long_text,
            **_split_settings(self.long_text, language, self.name),
            "bleurt_pytorch": version("bleurt-pytorch"),
        }

    def _model(self):
        """Load the checkpoint once; downloads it on first use."""

        def load():
            """Build model and tokenizer on the chosen device."""
            from bleurt_pytorch import (
                BleurtForSequenceClassification,
                BleurtTokenizer,
            )

            device = resolve_device(self.device)
            model = BleurtForSequenceClassification.from_pretrained(self.checkpoint)
            tokenizer = BleurtTokenizer.from_pretrained(self.checkpoint)
            return model.to(device).eval(), tokenizer, device

        return self._models.get(self.checkpoint, load)

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one note against its reference."""
        texts = _texts(prediction, sample)
        if texts is None:
            return None
        candidate, reference = texts
        model, tokenizer, device = self._model()
        score_pairs = partial(
            _regression_scores,
            model,
            tokenizer,
            device,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        value, details = _regression_note_score(
            score_pairs,
            reference=reference,
            candidate=candidate,
            language=sample.language,
            long_text=self.long_text,
            metric=self.name,
        )
        return Score(self.name, value, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average the scores; count truncated notes and, when split, pairs."""
        return _regression_aggregate(scores, self.long_text)


@dataclass(frozen=True)
class NegBleurt:
    """BLEURT fine-tuned to score negated statements low. English only.

    Anschütz, Miguel Lozano, and Groh, INLG 2023. They fine-tuned Google's
    BLEURT-tiny on their CANNOT sentence pairs, labeled 0 when negated and 1 when
    paraphrased, mixed with WMT ratings, and published it as a plain `transformers`
    BERT classifier. Its output is unbounded but mostly falls between 0 and 1. The
    loaded checkpoint stays with the configured metric until scoring ends.

    The paper scored single sentences. By default a note and its reference are one
    pair, truncated past `max_length` tokens; `long_text = "sentences"` scores
    sentence pairs instead. Datasets in other languages don't apply.
    """

    name: str = "negbleurt"
    group: str = "note_semantic"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))
    languages: tuple[str, ...] = ("en",)

    # Settings, from [metric_settings.negbleurt].
    checkpoint: str = "tum-nlp/NegBLEURT"
    device: str = "auto"
    # The checkpoint's tokenizer sets no limit; BLEURT-tiny was trained with 512.
    max_length: int = 512
    batch_size: int = 16
    long_text: str = TRUNCATE

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "NegBleurt":
        """Copy with settings from the experiment.

        Raises:
            ValueError: If a setting is unknown or `long_text` isn't one it takes.
        """
        allowed = ("checkpoint", "device", "max_length", "batch_size", "long_text")
        metric = apply_settings(self, settings, allowed)
        check_strategy(metric.name, metric.long_text, _SENTENCE_PAIR_STRATEGIES)
        return metric

    def describe(self, language: str) -> dict[str, object]:
        """Checkpoint, long-text strategy, and package versions.

        Raises:
            ValueError: If the extra is missing.
        """
        require_semantic("transformers", self.name)
        return {
            "checkpoint": self.checkpoint,
            "max_length": self.max_length,
            "long_text": self.long_text,
            **_split_settings(self.long_text, language, self.name),
            "transformers": version("transformers"),
        }

    def _model(self):
        """Load the checkpoint once; downloads it on first use."""

        def load():
            """Build model and tokenizer on the chosen device."""
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            device = resolve_device(self.device)
            model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
            tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            return model.to(device).eval(), tokenizer, device

        return self._models.get(self.checkpoint, load)

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Score one note against its reference, or return None outside English."""
        if sample.language not in self.languages:
            return None
        texts = _texts(prediction, sample)
        if texts is None:
            return None
        candidate, reference = texts
        model, tokenizer, device = self._model()
        score_pairs = partial(
            _regression_scores,
            model,
            tokenizer,
            device,
            max_length=self.max_length,
            batch_size=self.batch_size,
        )
        value, details = _regression_note_score(
            score_pairs,
            reference=reference,
            candidate=candidate,
            language=sample.language,
            long_text=self.long_text,
            metric=self.name,
        )
        return Score(self.name, value, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average the scores; count truncated notes and, when split, pairs."""
        return _regression_aggregate(scores, self.long_text)
