"""The telemetry scribe_eval understands, whatever pipeline produced it.

This is the contract for comparing workflows. The harness times each sample from the
outside, the same way for every pipeline. Pipelines add what only they can see from
the inside: phases, processes, and model usage. All of that is optional, and a metric
whose data is missing simply doesn't apply.

Phase names are shared across pipelines so they can be compared: `transcription` and
`note`, with `prefill` and `generation` nested under `note`.
"""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Phase:
    """A timed part of the work, measured inside the pipeline."""

    name: str
    seconds: float
    parent: str | None = None
    device: str | None = None  # "cpu", "accelerator", or None when unknown


@dataclass(frozen=True)
class ProcessUsage:
    """What one process consumed while handling a sample."""

    name: str
    peak_rss_bytes: int
    average_cpu_cores: float
    pid: int | None = None


@dataclass(frozen=True)
class ModelUsage:
    """Tokens and speed of one model call."""

    phase: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    prefill_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None


@dataclass(frozen=True)
class MemorySeries:
    """A process's memory over time, in seconds from the start of the sample."""

    process: str
    seconds: tuple[float, ...]
    bytes: tuple[int, ...]


@dataclass(frozen=True)
class SampleTelemetry:
    """Everything known about the cost of one sample."""

    wall_seconds: float = 0.0  # set by the harness
    phases: tuple[Phase, ...] = ()
    processes: tuple[ProcessUsage, ...] = ()
    models: tuple[ModelUsage, ...] = ()
    peak_rss_bytes: int | None = None
    trace_id: str | None = None
    run_id: str | None = None

    def to_dict(self) -> dict:
        """Serialize for a JSONL row."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleTelemetry":
        """Rebuild from a JSONL row."""
        return cls(
            wall_seconds=data["wall_seconds"],
            phases=tuple(Phase(**phase) for phase in data["phases"]),
            processes=tuple(ProcessUsage(**entry) for entry in data["processes"]),
            models=tuple(ModelUsage(**entry) for entry in data["models"]),
            peak_rss_bytes=data["peak_rss_bytes"],
            trace_id=data["trace_id"],
            run_id=data["run_id"],
        )
