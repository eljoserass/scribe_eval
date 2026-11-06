"""Who spoke when, scored with pyannote.metrics.

Needs the `diarization` extra. Both sides need times, so samples without them, like
TEME-v1's, don't apply.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from ..artifacts import DiarizedTranscript
from ..types import Prediction, Requires, Sample, Score
from .text import mean_values


def _annotation(transcript: DiarizedTranscript):
    """Turn timed speaker segments into pyannote's format."""
    from pyannote.core import Annotation
    from pyannote.core import Segment as Span

    annotation = Annotation()
    for index, segment in enumerate(transcript.segments):
        timed = segment.start is not None and segment.end is not None
        if timed and segment.speaker and segment.end > segment.start:
            annotation[Span(segment.start, segment.end), index] = segment.speaker
    return annotation


def _annotations(prediction: Prediction, sample: Sample):
    """Build both sides, or None if either has no timed speakers."""
    predicted = _annotation(prediction.artifacts["diarized_transcript"])
    reference = _annotation(sample.artifact("diarized_transcript"))
    return (reference, predicted) if len(reference) and len(predicted) else None


@dataclass(frozen=True)
class DiarizationErrorRate:
    """Share of speech time missed, invented, or given to the wrong speaker."""

    name: str = "der"
    group: str = "diarization"
    version: int = 1
    requires: Requires = Requires(("diarized_transcript",), ("diarized_transcript",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Keep the components, so the corpus rate is exact."""
        from pyannote.metrics.diarization import DiarizationErrorRate as Metric

        pair = _annotations(prediction, sample)
        if pair is None:
            return None
        parts = Metric()(*pair, detailed=True)
        details = {
            "confusion": float(parts["confusion"]),
            "false_alarm": float(parts["false alarm"]),
            "missed": float(parts["missed detection"]),
            "total": float(parts["total"]),
        }
        return Score(self.name, float(parts["diarization error rate"]), details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Weigh by speech time, and split the error by cause."""
        total = sum(score.details["total"] for score in scores)
        parts = {
            key: sum(score.details[key] for score in scores) / total
            for key in ("confusion", "false_alarm", "missed")
        }
        return {
            "global": sum(parts.values()),
            "sample_mean": fmean(score.value for score in scores),
            **parts,
        }


@dataclass(frozen=True)
class JaccardErrorRate:
    """DER's counterpart that gives every speaker equal weight."""

    name: str = "jer"
    group: str = "diarization"
    version: int = 1
    requires: Requires = Requires(("diarized_transcript",), ("diarized_transcript",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Compare speakers one by one."""
        from pyannote.metrics.diarization import JaccardErrorRate as Metric

        pair = _annotations(prediction, sample)
        return None if pair is None else Score(self.name, float(Metric()(*pair)))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average scores."""
        return mean_values(scores)
