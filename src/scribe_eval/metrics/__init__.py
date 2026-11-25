"""Registered metrics and their groups."""

from collections.abc import Iterable, Mapping

from ..judge import Asker
from ..types import Metric
from .clinical_bertscore import ClinicalBertScore
from .diarization import DiarizationErrorRate, JaccardErrorRate
from .grounding import Grounding
from .link_correct import LinkCorrect
from .note_format import CompressionRatio, FleschReadingEase, NoteWords
from .overlap import Bleu, Levenshtein, Meteor, Rouge
from .repetition import RepeatedLines, SequenceRepetition
from .resources import (
    ElapsedSeconds,
    PeakMemory,
    PhaseRealTimeFactor,
    PhaseSeconds,
    ProcessCpu,
    ProcessMemory,
    RealTimeFactor,
    Throughput,
    Tokens,
)
from .semantic import BertScore, Bleurt, NegBleurt
from .transcription import CHARACTER_ERROR_RATE, WORD_ERROR_RATE

_ALL: tuple[Metric, ...] = (
    WORD_ERROR_RATE,
    CHARACTER_ERROR_RATE,
    ClinicalBertScore(),
    Rouge("transcript_rouge_1", "rouge1", "transcript", "transcription"),
    Rouge("transcript_rouge_l", "rougeL", "transcript", "transcription"),
    Bleu("transcript_bleu", "transcript", "transcription"),
    SequenceRepetition("transcript_seq_rep", "transcript", "transcription"),
    DiarizationErrorRate(),
    JaccardErrorRate(),
    NoteWords(),
    FleschReadingEase(),
    CompressionRatio(),
    SequenceRepetition("seq_rep", "note", "note_format"),
    RepeatedLines(),
    Rouge("rouge_1", "rouge1", "note", "note_reference"),
    Rouge("rouge_2", "rouge2", "note", "note_reference"),
    Rouge("rouge_l", "rougeL", "note", "note_reference"),
    Bleu("bleu", "note", "note_reference"),
    Meteor(),
    Levenshtein(),
    BertScore(),
    Bleurt(),
    NegBleurt(),
    LinkCorrect("link"),
    LinkCorrect("correct"),
    Grounding(),
    ElapsedSeconds(),
    RealTimeFactor(),
    PeakMemory(),
    PhaseSeconds(),
    PhaseRealTimeFactor(),
    ProcessMemory(),
    ProcessCpu(),
    Throughput("prefill_tokens_per_second"),
    Throughput("generation_tokens_per_second"),
    Tokens("prompt_tokens"),
    Tokens("completion_tokens"),
)

METRICS: dict[str, Metric] = {metric.name: metric for metric in _ALL}
GROUPS: dict[str, tuple[Metric, ...]] = {}
for _metric in _ALL:
    GROUPS[_metric.group] = (*GROUPS.get(_metric.group, ()), _metric)


def needs_judge(metric: Metric) -> bool:
    """Tell whether a metric asks a language model, through `with_judge`."""
    return hasattr(metric, "with_judge")


def configure_metrics(
    settings: Mapping[str, Mapping[str, object]], *, judge: Asker | None
) -> dict[str, Metric]:
    """Apply an experiment's judge and `[metric_settings]`, once per experiment.

    Metrics that load models load them lazily, and datasets share them.

    Raises:
        ValueError: If settings name an unknown metric, a metric that takes none, or
            a setting the metric doesn't take.
    """
    unknown = sorted(set(settings) - set(METRICS))
    if unknown:
        raise ValueError(f"Settings for unknown metrics: {', '.join(unknown)}")
    configured = {}
    for name, metric in METRICS.items():
        if judge is not None and needs_judge(metric):
            metric = metric.with_judge(judge)
        if name in settings:
            if not hasattr(metric, "with_settings"):
                raise ValueError(f"Metric {name!r} takes no settings.")
            metric = metric.with_settings(settings[name])
        configured[name] = metric
    return configured


def resolve_metrics(
    names: Iterable[str], available: Mapping[str, Metric] | None = None
) -> tuple[Metric, ...]:
    """Expand metric and group names, keeping order and dropping duplicates.

    Args:
        names: Metric or group names.
        available: Configured metrics by name; the registry's defaults otherwise.

    Raises:
        ValueError: If a name is neither a metric nor a group.
    """
    available = METRICS if available is None else available
    resolved: dict[str, Metric] = {}
    for name in names:
        if name in GROUPS:
            resolved.update(
                {metric.name: available[metric.name] for metric in GROUPS[name]}
            )
        elif name in METRICS:
            resolved[name] = available[name]
        else:
            raise ValueError(f"Unknown metric or group: {name!r}")
    return tuple(resolved.values())
