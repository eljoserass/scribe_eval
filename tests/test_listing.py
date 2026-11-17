"""List runs from their saved records: filters, stale processes, failure reasons."""

import os
import socket
import tempfile
import unittest
from pathlib import Path

from scribe_eval.application import list_runs
from scribe_eval.listing import RunFilter
from scribe_eval.storage import append_jsonl, write_json

_DEAD_PID = 2**22 + 12345


def write_run(
    root: Path,
    name: str,
    *,
    state: str,
    started: str,
    done: int = 1,
    failed: int = 0,
    pid: int | None = None,
    error: str | None = None,
) -> Path:
    """Write the manifest and status of a fake run."""
    directory = root / f"20261117-120000-{name}-abc123"
    write_json(directory / "manifest.json", {"name": name, "created_at": started})
    status = {
        "state": state,
        "error": error,
        "updated_at": "2026-11-17T13:00:00+00:00",
        "datasets": {"tiny": {"selected": 2, "run": done, "failed": failed}},
    }
    if pid is not None:
        status.update(pid=pid, host=socket.gethostname())
    write_json(directory / "status.json", status)
    return directory


class ListingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def names(self, which: RunFilter) -> list[str]:
        """List run names for a filter."""
        return [summary.name for summary in list_runs(self.root, which=which)]

    def test_running_keeps_live_runs_and_marks_dead_ones_stale(self):
        write_run(
            self.root,
            "live",
            state="running",
            started="2026-11-17T10:00:00+00:00",
            pid=os.getpid(),
        )
        write_run(
            self.root,
            "dead",
            state="running",
            started="2026-11-17T11:00:00+00:00",
            pid=_DEAD_PID,
        )
        write_run(
            self.root,
            "done",
            state="completed",
            started="2026-11-17T09:00:00+00:00",
            done=2,
        )

        self.assertEqual(self.names(RunFilter.RUNNING), ["live"])
        states = {summary.name: summary.state for summary in list_runs(self.root)}
        self.assertEqual(states["dead"], "stale")
        self.assertIsNone(
            next(s for s in list_runs(self.root) if s.name == "live").ended_at
        )

    def test_all_is_newest_first_and_skips_other_directories(self):
        write_run(
            self.root, "old", state="completed", started="2026-11-16T10:00:00+00:00"
        )
        write_run(
            self.root, "new", state="completed", started="2026-11-17T10:00:00+00:00"
        )
        (self.root / "not-a-run").mkdir()

        self.assertEqual(self.names(RunFilter.ALL), ["new", "old"])

    def test_failed_explains_with_the_run_error_or_the_first_failed_sample(self):
        write_run(
            self.root,
            "crashed",
            state="failed",
            started="2026-11-17T10:00:00+00:00",
            error="Out of memory\ntraceback",
        )
        sample_failure = write_run(
            self.root,
            "partial",
            state="completed",
            started="2026-11-17T11:00:00+00:00",
            done=2,
            failed=1,
        )
        append_jsonl(
            sample_failure / "datasets" / "tiny" / "predictions.jsonl",
            {
                "variant": "clean",
                "sample_id": "b",
                "repeat": 0,
                "status": "error",
                "error": "Note rejected",
            },
        )
        write_run(
            self.root,
            "fine",
            state="completed",
            started="2026-11-17T12:00:00+00:00",
            done=2,
        )

        reasons = {
            s.name: s.reason for s in list_runs(self.root, which=RunFilter.FAILED)
        }
        self.assertEqual(
            reasons, {"partial": "tiny/b: Note rejected", "crashed": "Out of memory"}
        )

    def test_a_missing_runs_directory_lists_nothing(self):
        self.assertEqual(list_runs(self.root / "absent"), [])


if __name__ == "__main__":
    unittest.main()
