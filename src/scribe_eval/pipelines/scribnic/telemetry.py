"""Translate what Scribnic reports into the telemetry contract.

Scribnic's telemetry exists for client deployments and for detail inside the process.
Everything that depends on its field names, span names, and runtimes stays here.
"""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from scribnic import RunResult

from ...telemetry import MemorySeries, ModelUsage, Phase, ProcessUsage, SampleTelemetry
from ...telemetry.otlp import read_points, read_spans

_PHASE_BY_OUTPUT = {
    "transcript": "transcription",
    "diarized_transcript": "transcription",
    "note": "note",
}
_MEMORY_METRIC = "scribnic.process.memory.usage"
_RUN_ATTRIBUTE = "scribnic.run.id"
_STEP_SPAN = "scribnic.step.run"
_STEP_ATTRIBUTE = "scribnic.step.name"


def runtime_device(runtime: dict | None) -> str | None:
    """Tell where a runtime actually ran, from the command Scribnic started it with."""
    argv = (runtime or {}).get("argv")
    if not argv:
        return None
    if "--no-gpu" in argv:
        return "cpu"
    for flag in ("-ngl", "--device"):
        if flag in argv:
            value = argv[argv.index(flag) + 1]
            return "cpu" if value in ("0", "cpu") else "accelerator"
    return None


def _phase_names(result: RunResult) -> dict[str, str]:
    """Map a workflow's own step names to shared phase names."""
    names = {}
    for step in result.configuration.get("steps", []):
        output = step["model"]["output"]
        names[step["name"]] = _PHASE_BY_OUTPUT.get(output, step["name"])
    return names


def _phases(result: RunResult) -> tuple[Phase, ...]:
    """Convert steps, splitting a llama.cpp call into prefill and generation."""
    names = _phase_names(result)
    phases = []
    for step in result.steps:
        name = names.get(step.name, step.name)
        device = runtime_device(step.metadata.get("runtime"))
        phases.append(Phase(name, step.elapsed_seconds, device=device))
        timings = step.metadata.get("timings") or {}
        if "prompt_ms" in timings:
            seconds = timings["prompt_ms"] / 1000
            phases.append(Phase("prefill", seconds, parent=name, device=device))
        if "predicted_ms" in timings:
            seconds = timings["predicted_ms"] / 1000
            phases.append(Phase("generation", seconds, parent=name, device=device))
    return tuple(phases)


def _models(result: RunResult) -> tuple[ModelUsage, ...]:
    """Collect tokens and speed from the steps that report them."""
    names = _phase_names(result)
    models = []
    for step in result.steps:
        usage = step.metadata.get("usage") or {}
        timings = step.metadata.get("timings") or {}
        if usage or timings:
            models.append(
                ModelUsage(
                    names.get(step.name, step.name),
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    timings.get("prompt_per_second"),
                    timings.get("predicted_per_second"),
                )
            )
    return tuple(models)


def sample_telemetry(result: RunResult) -> SampleTelemetry:
    """Build the contract from one Scribnic result, complete or partial."""
    processes = tuple(
        ProcessUsage(
            entry["name"],
            entry["peak_sampled_rss_bytes"],
            entry["average_cpu_cores"],
            entry.get("pid"),
        )
        for entry in result.resources.get("processes", [])
    )
    return SampleTelemetry(
        phases=_phases(result),
        processes=processes,
        models=_models(result),
        peak_rss_bytes=result.resources.get("peak_sampled_process_rss_sum_bytes"),
        trace_id=result.trace_id,
        run_id=result.run_id,
    )


def memory_series(
    directories: Iterable[Path], telemetry: SampleTelemetry
) -> tuple[MemorySeries, ...]:
    """Read each process's memory during one sample, from its first point.

    Gauges keep being exported with their last value after a sample ends, so anything
    past the sample's own duration is dropped.
    """
    names = {str(entry.pid): entry.name for entry in telemetry.processes}
    grouped = defaultdict(list)
    for point in read_points(directories, _MEMORY_METRIC):
        if point.attributes.get(_RUN_ATTRIBUTE) == telemetry.run_id:
            pid = str(point.attributes.get("process.pid"))
            grouped[names.get(pid, pid)].append(point)
    if not grouped:
        return ()
    start = min(point.time_ns for points in grouped.values() for point in points)
    series = []
    for name, points in sorted(grouped.items()):
        points = sorted(points, key=lambda point: point.time_ns)
        kept = [
            p for p in points if (p.time_ns - start) / 1e9 <= telemetry.wall_seconds
        ]
        seconds = tuple((point.time_ns - start) / 1e9 for point in kept)
        values = tuple(int(point.value) for point in kept)
        series.append(MemorySeries(name, seconds, values))
    return tuple(series)


def phase_windows(
    directories: Iterable[Path], telemetry: SampleTelemetry
) -> tuple[tuple[str, float, float], ...]:
    """Locate each workflow step on the sample's timeline, from its spans."""
    spans = [s for s in read_spans(directories) if s.trace_id == telemetry.trace_id]
    if not spans:
        return ()
    start = min(span.start_ns for span in spans)
    return tuple(
        (
            str(span.attributes.get(_STEP_ATTRIBUTE)),
            (span.start_ns - start) / 1e9,
            (span.end_ns - start) / 1e9,
        )
        for span in sorted(spans, key=lambda span: span.start_ns)
        if span.name == _STEP_SPAN
    )
