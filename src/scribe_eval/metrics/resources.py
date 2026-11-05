"""Cost of a run, read from the telemetry contract."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, quantiles

from ..types import Prediction, Requires, Sample, Score
from .text import mean_values

_GROUP = "resources"
_AUDIO = "audio_seconds"


def _mean_by_key(scores: Sequence[Score]) -> dict[str, float]:
    """Average each detail across the scores that have it."""
    names = {name for score in scores for name in score.details}
    return {
        name: fmean(score.details[name] for score in scores if name in score.details)
        for name in sorted(names)
    }


@dataclass(frozen=True)
class ElapsedSeconds:
    """Time around the whole sample, measured by the harness for every pipeline."""

    name: str = "elapsed_seconds"
    group: str = _GROUP
    version: int = 2
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score:
        """Read the outside measurement."""
        return Score(self.name, prediction.telemetry.wall_seconds)

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
    group: str = _GROUP
    version: int = 2
    requires: Requires = Requires((), ("audio",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Compare time with audio length, when audio was the input."""
        if prediction.input_kind != "audio":
            return None
        seconds = float(sample.metadata["audio_seconds"])
        wall = prediction.telemetry.wall_seconds
        return Score(self.name, wall / seconds, {"wall": wall, "audio": seconds})

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Divide total time by total audio."""
        wall = sum(score.details["wall"] for score in scores)
        audio = sum(score.details["audio"] for score in scores)
        return {"global": wall / audio}


@dataclass(frozen=True)
class PhaseRealTimeFactor:
    """Each phase's time per second of audio, to see what actually scales with it."""

    name: str = "phase_real_time_factor"
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires((), ("audio",))

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Keep each phase's time next to the audio's length."""
        phases = {phase.name: phase.seconds for phase in prediction.telemetry.phases}
        if prediction.input_kind != "audio" or not phases:
            return None
        seconds = float(sample.metadata["audio_seconds"])
        return Score(
            self.name, sum(phases.values()) / seconds, {**phases, _AUDIO: seconds}
        )

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Divide each phase's total time by total audio."""
        audio = sum(score.details[_AUDIO] for score in scores)
        names = {name for score in scores for name in score.details} - {_AUDIO}
        return {
            name: sum(score.details.get(name, 0.0) for score in scores) / audio
            for name in sorted(names)
        }


@dataclass(frozen=True)
class PeakMemory:
    """Peak memory of the pipeline's processes together."""

    name: str = "peak_rss_bytes"
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Read the peak, if the pipeline reported one."""
        peak = prediction.telemetry.peak_rss_bytes
        return None if peak is None else Score(self.name, float(peak))

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the highest peak."""
        return {"max": max(score.value for score in scores)}


@dataclass(frozen=True)
class PhaseSeconds:
    """Time per phase, measured inside the pipeline."""

    name: str = "phase_seconds"
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Collect each phase's time."""
        phases = {phase.name: phase.seconds for phase in prediction.telemetry.phases}
        return Score(self.name, sum(phases.values()), phases) if phases else None

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the mean of each phase."""
        return _mean_by_key(scores)


@dataclass(frozen=True)
class ProcessMemory:
    """Peak memory of each process, to see what actually uses it."""

    name: str = "process_peak_rss_bytes"
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Collect each process's peak."""
        peaks = {
            entry.name: float(entry.peak_rss_bytes)
            for entry in prediction.telemetry.processes
        }
        return Score(self.name, max(peaks.values()), peaks) if peaks else None

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the highest peak of each process."""
        names = {name for score in scores for name in score.details}
        return {
            name: max(score.details.get(name, 0.0) for score in scores)
            for name in sorted(names)
        }


@dataclass(frozen=True)
class ProcessCpu:
    """Average CPU cores each process kept busy, idle time included."""

    name: str = "process_cpu_cores"
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Collect each process's average."""
        cores = {
            entry.name: entry.average_cpu_cores
            for entry in prediction.telemetry.processes
        }
        return Score(self.name, sum(cores.values()), cores) if cores else None

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report the mean of each process."""
        return _mean_by_key(scores)


@dataclass(frozen=True)
class Throughput:
    """Model speed in tokens per second, for pipelines that report it."""

    name: str
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Read the speed from the first model that reports it."""
        for model in prediction.telemetry.models:
            value = getattr(model, self.name)
            if value is not None:
                return Score(self.name, float(value))
        return None

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Average speeds."""
        return mean_values(scores)


@dataclass(frozen=True)
class Tokens:
    """Tokens used by the models. Not every pipeline reports them."""

    name: str
    group: str = _GROUP
    version: int = 1
    requires: Requires = Requires()

    def score(self, prediction: Prediction, sample: Sample) -> Score | None:
        """Add up tokens across models."""
        counts = [
            getattr(model, self.name)
            for model in prediction.telemetry.models
            if getattr(model, self.name) is not None
        ]
        return Score(self.name, float(sum(counts))) if counts else None

    def aggregate(self, scores: Sequence[Score]) -> dict[str, float]:
        """Report total and mean."""
        return {
            "total": sum(score.value for score in scores),
            "mean": fmean(score.value for score in scores),
        }
