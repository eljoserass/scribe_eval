"""Translate what Berta's worker and llama-server report into the telemetry contract.

The worker times Berta's two tasks. llama-server's cumulative counters, read before
and after a sample, split the note into prefill and generation. Memory and CPU come
from sampling the owned processes. Each sample's timeline is kept in a JSONL file in
the telemetry directory, so the report can draw it later.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ...storage import append_jsonl, read_jsonl
from ...telemetry import MemorySeries, ModelUsage, Phase, SampleTelemetry
from ..base import Timeline
from .sampling import ProcessSampler

TIMELINE_FILE = "berta-timeline.jsonl"


@dataclass(frozen=True)
class Devices:
    """Where each task runs: `cpu`, `accelerator`, or None when unknown."""

    transcription: str | None
    note: str | None


@dataclass(frozen=True)
class NoteTimings:
    """llama-server's own measurement of one note, from its counters."""

    prefill_seconds: float
    generation_seconds: float
    prefill_tokens_per_second: float | None
    generation_tokens_per_second: float | None


def note_timings(
    before: dict[str, float], after: dict[str, float]
) -> NoteTimings | None:
    """Measure what llama-server did between two counter snapshots.

    Prefill counts only prompt tokens the server evaluated, not those it reused
    from its cache, so its speed is evaluated tokens per second.

    Args:
        before: Counters read before the sample.
        after: Counters read after it.

    Returns:
        The timings, or None if the counters are missing or show no work.
    """
    names = (
        "prompt_tokens_total",
        "prompt_seconds_total",
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
    )
    if not all(name in before and name in after for name in names):
        return None
    prompt_tokens, prompt_seconds, predicted_tokens, predicted_seconds = (
        after[name] - before[name] for name in names
    )
    if prompt_seconds <= 0 and predicted_seconds <= 0:
        return None
    return NoteTimings(
        prompt_seconds,
        predicted_seconds,
        prompt_tokens / prompt_seconds if prompt_seconds > 0 else None,
        predicted_tokens / predicted_seconds if predicted_seconds > 0 else None,
    )


def _phases(
    reply: dict, timings: NoteTimings | None, devices: Devices
) -> tuple[Phase, ...]:
    """Name Berta's tasks with the shared phase names."""
    phases = []
    if "transcription_seconds" in reply:
        phases.append(
            Phase(
                "transcription",
                reply["transcription_seconds"],
                device=devices.transcription,
            )
        )
    if "note_seconds" in reply:
        phases.append(Phase("note", reply["note_seconds"], device=devices.note))
        if timings is not None:
            phases.append(
                Phase("prefill", timings.prefill_seconds, "note", device=devices.note)
            )
            phases.append(
                Phase(
                    "generation",
                    timings.generation_seconds,
                    "note",
                    device=devices.note,
                )
            )
    return tuple(phases)


def sample_telemetry(
    reply: dict,
    *,
    timings: NoteTimings | None,
    sampler: ProcessSampler,
    devices: Devices,
    run_id: str,
) -> SampleTelemetry:
    """Build the contract for one sample, complete or failed.

    Args:
        reply: The worker's reply, with each task's seconds and token counts.
        timings: llama-server's measurement of the note, if available.
        sampler: The sampler that ran during the sample, already stopped.
        devices: Where each task ran.
        run_id: This sample's identifier, linking it to its timeline row.
    """
    models = ()
    if "note" in reply:
        models = (
            ModelUsage(
                "note",
                reply.get("prompt_tokens"),
                reply.get("completion_tokens"),
                timings.prefill_tokens_per_second if timings else None,
                timings.generation_tokens_per_second if timings else None,
            ),
        )
    return SampleTelemetry(
        phases=_phases(reply, timings, devices),
        processes=sampler.usage(),
        models=models,
        peak_rss_bytes=sampler.peak_rss_bytes(),
        run_id=run_id,
    )


def _windows(reply: dict) -> list[tuple[str, float, float]]:
    """Place each task on the sample's timeline; the note follows transcription."""
    windows = []
    start = 0.0
    if "transcription_seconds" in reply:
        start = reply["transcription_seconds"]
        windows.append(("transcription", 0.0, start))
    if "note_seconds" in reply:
        windows.append(("note", start, start + reply["note_seconds"]))
    return windows


def save_timeline(
    directory: Path,
    reply: dict,
    *,
    sampler: ProcessSampler,
    run_id: str,
    started_at: float,
) -> None:
    """Append one sample's memory and task windows to the telemetry directory.

    Args:
        directory: The telemetry directory of the current pipeline opening.
        reply: The worker's reply.
        sampler: The stopped sampler for the sample.
        run_id: The sample's identifier.
        started_at: When the sample started, in Unix seconds.
    """
    memory = [
        {"process": series.process, "seconds": series.seconds, "bytes": series.bytes}
        for series in sampler.memory()
    ]
    row = {
        "run_id": run_id,
        "started_at": started_at,
        "windows": _windows(reply),
        "memory": memory,
    }
    append_jsonl(directory / TIMELINE_FILE, row)


def read_timeline(directories: Iterable[Path], telemetry: SampleTelemetry) -> Timeline:
    """Find one sample's timeline among the saved rows, or an empty one."""
    for directory in directories:
        for row in read_jsonl(directory / TIMELINE_FILE):
            if row["run_id"] != telemetry.run_id:
                continue
            memory = tuple(
                MemorySeries(
                    entry["process"], tuple(entry["seconds"]), tuple(entry["bytes"])
                )
                for entry in row["memory"]
            )
            windows = tuple((name, start, end) for name, start, end in row["windows"])
            return Timeline(memory, windows, row["started_at"])
    return Timeline()
