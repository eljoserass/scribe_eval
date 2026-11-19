"""Run Berta AI Scribe behind the `Pipeline` interface.

Berta can't share scribe_eval's process: it configures its services when imported,
from paths relative to its own directory, and pins dependencies that don't install
on every Python scribe_eval supports. So this pipeline runs Berta's own tasks in a
worker under Berta's environment, and serves the note model with llama-server.
"""

import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from uuid import uuid4

from ...artifacts import (
    Artifact,
    ArtifactKind,
    AudioFile,
    DiarizedTranscript,
    Note,
    Transcript,
)
from ...storage import write_json
from ...telemetry import SampleTelemetry
from ..base import PipelineResult, Timeline
from . import worker
from .processes import BertaWorker, LlamaServer, WorkerExited
from .sampling import ProcessSampler
from .telemetry import (
    Devices,
    note_timings,
    read_timeline,
    sample_telemetry,
    save_timeline,
)

WORKER_SCRIPT = Path(worker.__file__)

# llama-server runs as in Scribnic's `llama_cpp_prism` runtime and English workflow,
# so the note model is served the same way: one slot, 8192 tokens of context, six
# threads, accelerator offload on macOS only, no prompt cache across requests, and
# Bonsai's thinking turned off.
_CONTEXT_TOKENS = 8192
_THREADS = 6
_SERVER_OPTIONS = (
    "--metrics",
    "--perf",
    "--cache-ram",
    "0",
    "--chat-template-kwargs",
    '{"enable_thinking": false}',
)

# Fixed in Berta's llama.cpp adapter for every note request.
_BERTA_TEMPERATURE = 0
_BERTA_MAX_TOKENS = 4096

_PRODUCES: dict[ArtifactKind, set[ArtifactKind]] = {
    "audio": {"transcript", "note"},
    "transcript": {"note"},
    "diarized_transcript": {"note"},
}


def _git_commit(directory: Path) -> str | None:
    """Return the checked-out commit of a directory, or None if it has none."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or None


def _llama_server_version(binary: Path) -> str | None:
    """Ask llama-server for its version line, such as `version: 1 (build 2, ...)`."""
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in (result.stdout + result.stderr).splitlines():
        if line.startswith("version:"):
            return line.removeprefix("version:").strip()
    return None


def _existing(path: Path, description: str) -> Path:
    """Resolve a path that must exist.

    Raises:
        ValueError: If it doesn't.
    """
    if not path.exists():
        raise ValueError(f"{description} not found: {path}")
    return path.resolve()


def _existing_python(path: Path) -> Path:
    """Make an interpreter path absolute, keeping a virtual environment's symlink.

    Resolving the symlink would run the base interpreter, outside the environment.

    Raises:
        ValueError: If the interpreter doesn't exist.
    """
    if not path.exists():
        raise ValueError(f"Berta's Python not found: {path}")
    return Path(os.path.abspath(path))


class BertaPipeline:
    """Berta AI Scribe: its own transcription service, note prompt, and note type.

    Constructing it checks paths and reads Berta's commit. `prepare` starts
    llama-server and Berta's worker, which loads Whisper; both belong to the current
    `open` and stop when it ends, so `isolate_samples` leaves nothing behind. Logs go
    to the telemetry directory.
    """

    def __init__(
        self,
        *,
        base_directory: Path,
        berta_directory: str,
        llama_server: str,
        note_model: str,
        python: str | None = None,
        whisper_model: str = "small",
        whisper_compute_type: str = "float32",
        whisper_device: str = "cpu",
        note_type: str = "Full Visit",
    ) -> None:
        """Check the configuration without starting anything.

        Relative paths are resolved from the experiment's directory.

        Args:
            base_directory: The experiment's directory.
            berta_directory: Berta's `web-api` directory, holding the `app` package.
            llama_server: The llama-server binary that serves the note model.
            note_model: The GGUF file of the note model.
            python: Berta's Python; `.venv/bin/python` in `berta_directory` by
                default.
            whisper_model: The faster-whisper model Berta's WhisperX service loads,
                in place of Berta's hardcoded `large-v3`.
            whisper_compute_type: CTranslate2 compute type; Berta uses `float32`.
            whisper_device: Berta's `WHISPERX_DEVICE`: `cpu`, `cuda`, or `cuda:N`.
            note_type: The title of one of Berta's built-in note types.

        Raises:
            ValueError: If a path doesn't exist.
        """
        self.berta_directory = _existing(
            base_directory / berta_directory, "Berta's web-api directory"
        )
        _existing(self.berta_directory / "app" / "tasks", "Berta's app package")
        default_python = self.berta_directory / ".venv" / "bin" / "python"
        self.python = _existing_python(
            base_directory / python if python else default_python
        )
        self.llama_server = _existing(base_directory / llama_server, "llama-server")
        self.note_model = _existing(base_directory / note_model, "The note model")
        self.whisper_model = whisper_model
        self.whisper_compute_type = whisper_compute_type
        self.whisper_device = whisper_device
        self.note_type = note_type
        self.commit = _git_commit(self.berta_directory)
        self._telemetry_directory: Path | None = None
        self._data_directory: Path | None = None
        self._server: LlamaServer | None = None
        self._worker: BertaWorker | None = None

    @property
    def _served_model(self) -> str:
        """The name llama-server serves the note model under."""
        return self.note_model.stem

    @property
    def _gpu_layers(self) -> str:
        """Offload every layer on macOS and none elsewhere, as Scribnic does."""
        return "all" if platform.system() == "Darwin" else "0"

    def _server_arguments(self) -> list[str]:
        """Build llama-server's command, without host and port."""
        return [
            str(self.llama_server),
            "-m",
            str(self.note_model),
            "--alias",
            self._served_model,
            "-c",
            str(_CONTEXT_TOKENS),
            "-np",
            "1",
            "-ngl",
            self._gpu_layers,
            "-t",
            str(_THREADS),
            "-tb",
            str(_THREADS),
            *_SERVER_OPTIONS,
        ]

    def describe(self) -> dict[str, object]:
        """Return Berta's effective configuration, with both models and runtimes."""
        return {
            "name": "berta",
            "berta_directory": str(self.berta_directory),
            "berta_commit": self.commit,
            "python": str(self.python),
            "transcription": {
                "service": "WhisperX",
                "model": self.whisper_model,
                "compute_type": self.whisper_compute_type,
                "device": self.whisper_device,
                "audio": "converted to MP3 at Berta's default bitrate, as on upload",
            },
            "note": {
                "service": "LlamaCpp",
                "note_type": self.note_type,
                "output_type": worker.OUTPUT_TYPE,
                "temperature": _BERTA_TEMPERATURE,
                "max_tokens": _BERTA_MAX_TOKENS,
                "model": self._served_model,
                "llama_server": self._server_arguments(),
            },
        }

    def produces(self, input_kind: ArtifactKind) -> set[ArtifactKind]:
        """Make a transcript and a note from audio, or a note from a transcript.

        Raises:
            ValueError: For a note as input.
        """
        if input_kind not in _PRODUCES:
            raise ValueError(f"Berta does not accept {input_kind!r} as input.")
        return set(_PRODUCES[input_kind])

    @contextmanager
    def open(self, telemetry_directory: Path) -> Iterator[None]:
        """Own the processes `prepare` starts, and stop them on exit.

        Berta's recordings folder and database file point to a temporary
        directory, deleted on exit, so Berta's checkout stays untouched.
        """
        with ExitStack() as stack:
            self._telemetry_directory = telemetry_directory
            self._data_directory = Path(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="berta-"))
            )
            stack.callback(self._release)
            yield

    def _release(self) -> None:
        """Stop the worker, then the server it uses."""
        try:
            if self._worker is not None:
                self._worker.close()
        finally:
            self._worker = None
            if self._server is not None:
                self._server.close()
            self._server = None

    def _worker_environment(self, server_url: str) -> dict[str, str]:
        """Configure Berta through its settings, as its `.env` file would.

        Berta also loads `.env` from its directory, but these values take
        precedence. No cloud service is configured, and model downloads are off:
        everything runs locally from cached models.
        """
        inherited = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith(("PYTHON", "VIRTUAL_ENV"))
        }
        return inherited | {
            "PYTHONPATH": str(self.berta_directory),
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "ENVIRONMENT": "development",
            "USE_AURORA": "false",
            "USE_COGNITO": "false",
            "USE_GOOGLE_AUTH": "false",
            "ACCESS_TOKEN_SECRET": uuid4().hex,
            "ACCESS_TOKEN_EXPIRE_MINUTES": "1",
            "RECORDINGS_FOLDER": str(self._data_directory / "recordings"),
            "DEV_DATABASE_FILE": str(self._data_directory / "database.db"),
            "TRANSCRIPTION_SERVICE": "WhisperX",
            "WHISPERX_DEVICE": self.whisper_device,
            "GENERATIVE_AI_SERVICE": "LlamaCpp",
            "LLAMA_CPP_SERVER_URL": server_url,
            "DEFAULT_NOTE_GENERATION_MODEL": self._served_model,
            worker.WHISPER_MODEL_VARIABLE: self.whisper_model,
            worker.WHISPER_COMPUTE_TYPE_VARIABLE: self.whisper_compute_type,
            worker.NOTE_MODEL_VARIABLE: self._served_model,
            worker.NOTE_TYPE_VARIABLE: self.note_type,
        }

    def prepare(self, input_kind: ArtifactKind) -> dict[str, object]:
        """Start llama-server, then Berta's worker, unless they already run.

        Returns:
            Load times, Berta's versions and services, and llama-server's version.

        Raises:
            RuntimeError: If either process fails to start.
        """
        if self._worker is not None:
            return {"elapsed_seconds": 0.0, "already_loaded": True}
        # Each process is kept only once it started, so a failure leaves nothing
        # half-open; a server started before a failing worker stops with `open`.
        started = time.perf_counter()
        server = LlamaServer(
            self._server_arguments(),
            log_path=self._telemetry_directory / "llama-server.log",
        )
        server.start()
        self._server = server
        server_seconds = time.perf_counter() - started
        berta = BertaWorker(
            [str(self.python), str(WORKER_SCRIPT)],
            directory=self.berta_directory,
            environment=self._worker_environment(server.url),
            log_path=self._telemetry_directory / "berta-worker.log",
        )
        ready = berta.start()
        self._worker = berta
        loaded = {k: v for k, v in ready.items() if k not in ("status", "load_seconds")}
        return {
            "elapsed_seconds": time.perf_counter() - started,
            "llama_server_seconds": server_seconds,
            "berta_seconds": ready["load_seconds"],
            "llama_server_version": _llama_server_version(self.llama_server),
            **loaded,
        }

    def _devices(self) -> Devices:
        """Tell where each task runs, from the settings each runtime received."""
        transcription = "cpu" if self.whisper_device == "cpu" else "accelerator"
        note = "cpu" if self._gpu_layers == "0" else "accelerator"
        return Devices(transcription, note)

    def run(self, source: Artifact, output_directory: Path) -> PipelineResult:
        """Run one input through Berta. A failed task is a result.

        Berta's reply, with its timings and token counts, is kept as `berta.json`
        in `output_directory`.

        Raises:
            RuntimeError: If a process stopped before this sample, which would make
                every remaining sample fail. Reopening the pipeline restarts them.
        """
        if self._worker is None or self._server is None:
            raise RuntimeError("Call prepare before run.")
        if not (self._worker.running() and self._server.running()):
            raise RuntimeError(
                f"Berta's processes stopped. See {self._telemetry_directory}"
            )
        request = _request(source)
        run_id = uuid4().hex
        started_at = time.time()
        before = self._server.counters()
        process_ids = {
            "llama-server": self._server.process.pid,
            "berta-worker": self._worker.process.pid,
        }
        with ProcessSampler(process_ids) as sampler:
            try:
                reply = self._worker.request(request)
            except WorkerExited as error:
                reply = {"status": "error", "error": str(error)}
        timings = note_timings(before, self._server.counters())
        output_directory.mkdir(parents=True, exist_ok=True)
        write_json(output_directory / "berta.json", reply)
        save_timeline(
            self._telemetry_directory,
            reply,
            sampler=sampler,
            run_id=run_id,
            started_at=started_at,
        )
        telemetry = sample_telemetry(
            reply,
            timings=timings,
            sampler=sampler,
            devices=self._devices(),
            run_id=run_id,
        )
        return PipelineResult(
            status="success" if reply["status"] == "success" else "error",
            artifacts=_artifacts(reply),
            error=reply.get("error"),
            telemetry=telemetry,
        )

    def timeline(
        self, telemetry_directories: Iterable[Path], telemetry: SampleTelemetry
    ) -> Timeline:
        """Read the sample's memory and task windows saved during the run."""
        return read_timeline(telemetry_directories, telemetry)


def _request(source: Artifact) -> dict[str, str]:
    """Build the worker's request; Berta's transcripts carry no speakers.

    Audio paths are made absolute because the worker runs in Berta's `web-api`
    directory, where a path relative to the harness would not exist.
    """
    if isinstance(source, AudioFile):
        return {"audio": str(Path(source.path).resolve())}
    if isinstance(source, DiarizedTranscript):
        return {"transcript": source.without_speakers().text}
    if isinstance(source, Transcript):
        return {"transcript": source.text}
    raise ValueError(f"Berta does not accept {source.kind!r} as input.")


def _artifacts(reply: dict) -> dict[ArtifactKind, Artifact]:
    """Keep what Berta produced, never its input."""
    artifacts = {}
    if reply.get("transcript") is not None:
        artifacts["transcript"] = Transcript.from_text(reply["transcript"])
    if reply.get("note") is not None:
        artifacts["note"] = Note(reply["note"])
    return artifacts
