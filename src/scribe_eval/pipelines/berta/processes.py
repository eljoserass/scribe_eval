"""The processes the Berta pipeline starts, owns, and stops.

Two processes serve a run: llama-server, which serves the note model, and Berta's
worker, which runs under Berta's own Python. Each writes its output to a log file,
never to a pipe nobody reads, so neither can block on a full buffer.
"""

import json
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import IO

_STOP_SECONDS = 10.0
_KILL_SECONDS = 5.0
_LOG_TAIL_CHARACTERS = 2000


def _free_port() -> int:
    """Pick a free loopback TCP port for a server this pipeline owns."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _log_tail(path: Path) -> str:
    """Read the end of a process log, for an error message."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-_LOG_TAIL_CHARACTERS:]


def _stop(process: subprocess.Popen | None) -> None:
    """Terminate a process if it still runs, killing it if it won't stop."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_STOP_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=_KILL_SECONDS)


def parse_metrics(text: str) -> dict[str, float]:
    """Read llama-server's Prometheus counters, without the `llamacpp:` prefix.

    Args:
        text: The body of llama-server's `/metrics` endpoint.

    Returns:
        Each sample's value by name, such as `prompt_seconds_total`.
    """
    counters = {}
    for line in text.splitlines():
        if line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        try:
            counters[name.removeprefix("llamacpp:")] = float(value)
        except ValueError:
            continue
    return counters


class LlamaServer:
    """One llama-server process serving the note model, owned until `close`.

    The server listens on a free loopback port and writes its log to `log_path`.
    """

    def __init__(
        self,
        arguments: list[str],
        *,
        log_path: Path,
        startup_timeout_seconds: float = 120.0,
    ) -> None:
        """Keep the command without starting anything.

        Args:
            arguments: The binary and its options, without `--host` or `--port`.
            log_path: Where the server's output goes; overwritten on start.
            startup_timeout_seconds: How long to wait for the model to load.
        """
        self.arguments = arguments
        self.log_path = log_path
        self.startup_timeout_seconds = startup_timeout_seconds
        self.process: subprocess.Popen | None = None
        self.url = ""
        self._log: IO[str] | None = None

    def start(self) -> None:
        """Start the server and wait until it reports ready.

        Raises:
            RuntimeError: If the server exits or doesn't get ready in time. The
                process is stopped before raising.
        """
        port = _free_port()
        self.url = f"http://127.0.0.1:{port}"
        command = [*self.arguments, "--host", "127.0.0.1", "--port", str(port)]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command,
                cwd=Path(self.arguments[0]).parent,
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=subprocess.STDOUT,
            )
            self._wait_until_ready()
        except BaseException:
            self.close()
            raise

    def _wait_until_ready(self) -> None:
        """Poll `/health` until the model is loaded.

        Raises:
            RuntimeError: If the process exits or the timeout passes.
        """
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited during startup:\n{_log_tail(self.log_path)}"
                )
            if self._healthy():
                return
            time.sleep(0.2)
        raise RuntimeError(f"llama-server wasn't ready in time. See {self.log_path}")

    def _healthy(self) -> bool:
        """Tell whether the server answers `/health` with status ok."""
        try:
            with urllib.request.urlopen(f"{self.url}/health", timeout=1) as response:
                return json.load(response).get("status") == "ok"
        except (OSError, ValueError):
            return False

    def counters(self) -> dict[str, float]:
        """Read the server's cumulative counters, or nothing if unavailable."""
        try:
            with urllib.request.urlopen(f"{self.url}/metrics", timeout=2) as response:
                return parse_metrics(response.read().decode("utf-8"))
        except OSError:
            return {}

    def running(self) -> bool:
        """Tell whether the process is alive."""
        return self.process is not None and self.process.poll() is None

    def close(self) -> None:
        """Stop the process and close its log. Safe to call more than once."""
        try:
            _stop(self.process)
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None


class WorkerExited(RuntimeError):
    """Berta's worker stopped answering because its process ended."""


class BertaWorker:
    """Berta's worker process, which holds Berta's models until `close`.

    It speaks one JSON object per line over stdin and stdout; its stderr goes to
    `log_path`. Requests are sequential, so at most one reply is ever pending.
    """

    def __init__(
        self,
        command: list[str],
        *,
        directory: Path,
        environment: dict[str, str],
        log_path: Path,
    ) -> None:
        """Keep the command without starting anything.

        Args:
            command: Berta's Python and the worker script.
            directory: Berta's `web-api` directory, which Berta's paths assume.
            environment: The complete environment, including Berta's settings.
            log_path: Where the worker's stderr goes; overwritten on start.
        """
        self.command = command
        self.directory = directory
        self.environment = environment
        self.log_path = log_path
        self.process: subprocess.Popen | None = None
        self._log: IO[str] | None = None

    def start(self) -> dict[str, object]:
        """Start the worker and wait while it loads Berta.

        Returns:
            The worker's ready message: load time, versions, and services.

        Raises:
            RuntimeError: If Berta fails to load. The process is stopped first.
        """
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.command,
                cwd=self.directory,
                env=self.environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self._log,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
            message = self._receive()
            if message.get("status") != "ready":
                raise RuntimeError(f"Berta failed to load: {message.get('error')}")
            return message
        except WorkerExited as error:
            self.close()
            raise RuntimeError(f"{error}\n{_log_tail(self.log_path)}") from error
        except BaseException:
            self.close()
            raise

    def request(self, message: dict[str, object]) -> dict[str, object]:
        """Send one request and wait for its reply.

        Raises:
            WorkerExited: If the process ends before replying.
        """
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise WorkerExited(self._exit_message()) from error
        return self._receive()

    def _receive(self) -> dict[str, object]:
        """Read one reply line.

        Raises:
            WorkerExited: If stdout closes first.
        """
        line = self.process.stdout.readline()
        if not line:
            try:
                self.process.wait(timeout=_STOP_SECONDS)
            except subprocess.TimeoutExpired:
                pass
            raise WorkerExited(self._exit_message())
        return json.loads(line)

    def _exit_message(self) -> str:
        """Explain that the worker ended, pointing to its log."""
        code = self.process.poll()
        return f"Berta's worker exited with code {code}. See {self.log_path}"

    def running(self) -> bool:
        """Tell whether the process is alive."""
        return self.process is not None and self.process.poll() is None

    def close(self) -> None:
        """Let the worker finish by closing stdin, then make sure it ends.

        Safe to call more than once.
        """
        try:
            if self.process is not None and self.process.stdin is not None:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=_STOP_SECONDS)
                except subprocess.TimeoutExpired:
                    _stop(self.process)
            if self.process is not None and self.process.stdout is not None:
                self.process.stdout.close()
        finally:
            if self._log is not None:
                self._log.close()
                self._log = None
