"""Note similarity from neural models: BERTScore, BLEURT, and NegBLEURT.

Zhang et al., "BERTScore: Evaluating Text Generation with BERT", ICLR 2020.
https://arxiv.org/abs/1904.09675
Sellam et al., "BLEURT: Learning Robust Metrics for Text Generation", ACL 2020.
https://arxiv.org/abs/2004.04696
Anschütz, Miguel Lozano, and Groh, "This is not correct! Negation-aware Evaluation
of Language Generation Systems", INLG 2023.
https://aclanthology.org/2023.inlg-main.12/

All need the `semantic` extra and download their weights on first use. Inputs longer
than the model's limit, 512 tokens for the defaults, are truncated; `truncated` in
the details flags it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from statistics import fmean
from typing import TYPE_CHECKING

from ..types import Prediction, Requires, Sample, Score
from .models import LoadedModels, apply_settings, require_semantic, resolve_device

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


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


def _regression_score(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    device: str,
    *,
    reference: str,
    candidate: str,
    max_length: int,
) -> tuple[float, bool]:
    """Score one pair with a BLEURT-style model: BERT plus a regression head.

    The pair is one input, reference first, as in BLEURT. Past `max_length` tokens,
    the longer text loses tokens first, as BLEURT's own encoder does.

    Args:
        model: Sequence classifier with a single output, in evaluation mode.
        tokenizer: The checkpoint's tokenizer.
        device: Where `model` is.
        reference: The reference note.
        candidate: The generated note.
        max_length: Token limit for the pair, special tokens included.

    Returns:
        The model's output, and whether the pair was truncated.
    """
    import torch

    inputs = tokenizer(
        [reference],
        [candidate],
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        value = float(model(**inputs).logits.flatten()[0])
    length = len(tokenizer(reference, candidate)["input_ids"])
    return value, length > max_length


@dataclass(frozen=True)
class BertScore:
    """F1 of token embeddings matched greedily by cosine similarity.

    Zhang et al., ICLR 2020, through their `bert-score` package. Without a `model`,
    it uses the package's default for the sample's language: `roberta-large` in
    English, multilingual BERT otherwise. `describe` gives what to report.
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

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "BertScore":
        """Copy with settings from the experiment."""
        allowed = (
            "model",
            "num_layers",
            "rescale_with_baseline",
            "device",
            "batch_size",
        )
        return apply_settings(self, settings, allowed)

    def describe(self, language: str) -> dict[str, object]:
        """Effective model and layer, the parts of BERTScore's own hash."""
        require_semantic("bert_score", self.name)
        from bert_score.utils import lang2model, model2layers

        model = self.model or lang2model[language]
        return {
            "model": model,
            "num_layers": self.num_layers or model2layers.get(model),
            "rescale_with_baseline": self.rescale_with_baseline,
            "bert_score": version("bert-score"),
        }

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
        precision, recall, f1 = scorer.score(
            [candidate], [reference], batch_size=self.batch_size
        )
        limit = tokenizer.model_max_length
        longest = max(len(tokenizer.encode(text)) for text in texts)
        details = {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "truncated": float(longest > limit),
        }
        return Score(self.name, float(f1[0]), details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average F1, precision, and recall."""
        return _mean_details(scores, ("precision", "recall"))


@dataclass(frozen=True)
class Bleurt:
    """A learned score: BERT fine-tuned to predict human ratings of similarity.

    Sellam et al., ACL 2020. Google's code needs TensorFlow; this uses the
    `bleurt-pytorch` port and its conversions of the official checkpoints.
    BLEURT-20 is Google's recommended checkpoint; the D12, D6, and D3 distillations
    are smaller and faster. Scores roughly span 0 to 1, and aren't comparable across
    checkpoints.
    """

    name: str = "bleurt"
    group: str = "note_semantic"
    version: int = 1
    requires: Requires = Requires(("note",), ("note",))

    # Settings, from [metric_settings.bleurt].
    checkpoint: str = "lucadiliello/BLEURT-20"
    device: str = "auto"
    max_length: int = 512

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "Bleurt":
        """Copy with settings from the experiment."""
        return apply_settings(self, settings, ("checkpoint", "device", "max_length"))

    def describe(self, language: str) -> dict[str, object]:
        """Checkpoint and package version."""
        require_semantic("bleurt_pytorch", self.name)
        return {
            "checkpoint": self.checkpoint,
            "max_length": self.max_length,
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
        value, truncated = _regression_score(
            model,
            tokenizer,
            device,
            reference=reference,
            candidate=candidate,
            max_length=self.max_length,
        )
        return Score(self.name, value, {"truncated": float(truncated)})

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average the scores."""
        return _mean_details(scores, ())


@dataclass(frozen=True)
class NegBleurt:
    """BLEURT fine-tuned to score negated statements low. English only.

    Anschütz, Miguel Lozano, and Groh, INLG 2023. They fine-tuned Google's
    BLEURT-tiny on their CANNOT sentence pairs, labeled 0 when negated and 1 when
    paraphrased, mixed with WMT ratings, and published it as a plain `transformers`
    BERT classifier. Its output is unbounded but mostly falls between 0 and 1.

    The paper scored single sentences. Here a note and its reference are one pair,
    truncated past `max_length` tokens. Datasets in other languages don't apply.
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

    _models: LoadedModels = field(
        init=False, default_factory=LoadedModels, compare=False, repr=False
    )

    def with_settings(self, settings: Mapping[str, object]) -> "NegBleurt":
        """Copy with settings from the experiment."""
        return apply_settings(self, settings, ("checkpoint", "device", "max_length"))

    def describe(self, language: str) -> dict[str, object]:
        """Checkpoint and package version."""
        require_semantic("transformers", self.name)
        return {
            "checkpoint": self.checkpoint,
            "max_length": self.max_length,
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
        value, truncated = _regression_score(
            model,
            tokenizer,
            device,
            reference=reference,
            candidate=candidate,
            max_length=self.max_length,
        )
        return Score(self.name, value, {"truncated": float(truncated)})

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average the scores."""
        return _mean_details(scores, ())
