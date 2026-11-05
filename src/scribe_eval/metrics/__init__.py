"""Registered metrics and their groups."""

from collections.abc import Iterable

from ..types import Metric
from .note_format import NoteWords, SoapSections
from .note_reference import RougeL
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
from .transcription import CHARACTER_ERROR_RATE, WORD_ERROR_RATE

_ALL: tuple[Metric, ...] = (
    WORD_ERROR_RATE,
    CHARACTER_ERROR_RATE,
    SoapSections(),
    NoteWords(),
    RougeL(),
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


def resolve_metrics(names: Iterable[str]) -> tuple[Metric, ...]:
    """Expand metric and group names, keeping order and dropping duplicates.

    Raises:
        ValueError: If a name is neither a metric nor a group.
    """
    resolved: dict[str, Metric] = {}
    for name in names:
        if name in GROUPS:
            resolved.update({metric.name: metric for metric in GROUPS[name]})
        elif name in METRICS:
            resolved[name] = METRICS[name]
        else:
            raise ValueError(f"Unknown metric or group: {name!r}")
    return tuple(resolved.values())
