"""Cost of a run, read from what Scribnic reported."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, quantiles

from ..types import Prediction, Requires, Sample, Score
from .text import mean_values


@dataclass(frozen=True)
class ElapsedSeconds:
    """Time spent in workflow steps, excluding model loading done at startup."""

    name: str = "elapsed_seconds"
    group: str = "resources"
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Read the recorded time."""
        return Score(self.name, prediction.elapsed_seconds)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report mean and, with enough samples, the 95th percentile."""
        values = [score.value for score in scores]
        result = mean_values(scores)
        if len(values) >= 2:
            result["p95"] = quantiles(values, n=20, method="inclusive")[-1]
        return result


@dataclass(frozen=True)
class RealTimeFactor:
    """Processing time per second of audio; below 1 is faster than real time."""

    name: str = "real_time_factor"
    group: str = "resources"
    version: int = 1
    requires: Requires = Requires((), ("audio",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Compare time with audio length, when audio was the input."""
        if prediction.input_kind != "audio":
            return None
        seconds = float(sample.metadata["audio_seconds"])
        details = {"processing_seconds": prediction.elapsed_seconds, "audio": seconds}
        return Score(self.name, prediction.elapsed_seconds / seconds, details)

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Divide total time by total audio."""
        processing = sum(score.details["processing_seconds"] for score in scores)
        audio = sum(score.details["audio"] for score in scores)
        return {"global": processing / audio}


@dataclass(frozen=True)
class PeakMemory:
    """Peak sampled RSS, adding the client and its runtimes."""

    name: str = "peak_rss_bytes"
    group: str = "resources"
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Read the recorded peak, if the run reported one."""
        peak = prediction.resources.get("peak_sampled_process_rss_sum_bytes")
        return None if peak is None else Score(self.name, float(peak))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the highest peak."""
        return {"max": max(score.value for score in scores)}


def _token_total(prediction: Prediction, field: str) -> int | None:
    """Add a usage field across steps; None if no step reports it."""
    counts = [
        step["usage"][field]
        for step in prediction.steps
        if step.get("usage") and field in step["usage"]
    ]
    return sum(counts) if counts else None


@dataclass(frozen=True)
class Tokens:
    """Tokens reported by the backends. Not all of them report usage."""

    name: str
    group: str = "resources"
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Read the token count, if any step reported it."""
        total = _token_total(prediction, self.name)
        return None if total is None else Score(self.name, float(total))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report total and mean."""
        return {
            "total": sum(score.value for score in scores),
            "mean": fmean(score.value for score in scores),
        }
