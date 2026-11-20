"""Measure the memory and CPU of the processes the Berta pipeline owns."""

import threading
import time
from types import TracebackType

import psutil

from ...telemetry import MemorySeries, ProcessUsage


def _cpu_seconds(process: psutil.Process) -> float:
    """Total CPU time of a process and its finished children, like ffmpeg."""
    times = process.cpu_times()
    return times.user + times.system + times.children_user + times.children_system


class ProcessSampler:
    """Sample owned processes on a background thread for the length of one sample.

    Use it as a context manager around one sample. The thread starts on entry and
    stops, joined, on exit. It holds no process handles beyond that.
    """

    def __init__(
        self,
        process_ids: dict[str, int],
        *,
        interval_seconds: float = 0.25,
    ) -> None:
        """Choose what to sample.

        Args:
            process_ids: Each process's PID by its display name.
            interval_seconds: Time between memory samples.
        """
        self.interval_seconds = interval_seconds
        self._processes = {
            name: psutil.Process(pid) for name, pid in process_ids.items()
        }
        self._cpu_start: dict[str, float] = {}
        self._cpu_end: dict[str, float] = {}
        self._samples: dict[str, list[tuple[float, int]]] = {
            name: [] for name in process_ids
        }
        self._sums: list[int] = []
        self._started = 0.0
        self._elapsed = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_until_stopped, daemon=True)

    def __enter__(self) -> "ProcessSampler":
        """Record CPU times and start sampling memory."""
        self._started = time.monotonic()
        self._cpu_start = self._cpu_now()
        self._thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop sampling, take a last sample, and record CPU times."""
        self._stop.set()
        self._thread.join()
        self._sample_once()
        self._elapsed = time.monotonic() - self._started
        self._cpu_end = self._cpu_now()

    def _cpu_now(self) -> dict[str, float]:
        """Read each live process's CPU seconds."""
        seconds = {}
        for name, process in self._processes.items():
            try:
                seconds[name] = _cpu_seconds(process)
            except psutil.Error:
                continue
        return seconds

    def _sample_until_stopped(self) -> None:
        """Sample memory at each interval until told to stop."""
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_seconds)

    def _sample_once(self) -> None:
        """Record each live process's resident memory, and their sum."""
        seconds = time.monotonic() - self._started
        total = 0
        for name, process in self._processes.items():
            try:
                rss = process.memory_info().rss
            except psutil.Error:
                continue
            self._samples[name].append((seconds, rss))
            total += rss
        self._sums.append(total)

    def usage(self) -> tuple[ProcessUsage, ...]:
        """Report each process's peak memory and average CPU cores."""
        usage = []
        for name, process in self._processes.items():
            samples = self._samples[name]
            if not samples or name not in self._cpu_end:
                continue
            cpu_seconds = self._cpu_end[name] - self._cpu_start.get(name, 0.0)
            cores = cpu_seconds / self._elapsed if self._elapsed > 0 else 0.0
            peak = max(rss for _, rss in samples)
            usage.append(ProcessUsage(name, peak, cores, process.pid))
        return tuple(usage)

    def peak_rss_bytes(self) -> int | None:
        """The highest sampled sum of all processes' resident memory."""
        return max(self._sums) if self._sums else None

    def memory(self) -> tuple[MemorySeries, ...]:
        """Each process's memory over the sample, in seconds from its start."""
        return tuple(
            MemorySeries(
                name,
                tuple(seconds for seconds, _ in samples),
                tuple(rss for _, rss in samples),
            )
            for name, samples in self._samples.items()
            if samples
        )
