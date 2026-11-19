"""Run Berta AI Scribe's own tasks inside Berta's environment.

`BertaPipeline` starts this script with the Python of Berta's virtual environment,
in Berta's `web-api` directory, with Berta's settings in the environment. It never
runs inside scribe_eval's process: Berta configures itself at import time, with
module-level services, and needs Python and dependency versions of its own.

The protocol is one JSON object per line. The first line on stdout is either
`{"status": "ready", ...}` or `{"status": "failed", "error": ...}`. Each request on
stdin, `{"audio": path}` or `{"transcript": text}`, gets one reply. The worker exits
when stdin closes. Anything else printed, by Berta or its libraries, goes to stderr.

This file imports only the standard library and Berta, so it must stay compatible
with the Python that Berta's environment uses (3.11).
"""

import asyncio
import json
import os
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import TextIO

WHISPER_MODEL_VARIABLE = "SCRIBE_EVAL_WHISPER_MODEL"
WHISPER_COMPUTE_TYPE_VARIABLE = "SCRIBE_EVAL_WHISPER_COMPUTE_TYPE"
NOTE_MODEL_VARIABLE = "SCRIBE_EVAL_NOTE_MODEL"
NOTE_TYPE_VARIABLE = "SCRIBE_EVAL_NOTE_TYPE"

# Berta's built-in note types are stored as Markdown, but each note type can choose
# plain text. Markdown symbols would count as words in overlap metrics.
OUTPUT_TYPE = "Plain Text"

# Berta's web app sends no context for a note unless the clinician types some.
NOTE_CONTEXT = None

_PACKAGES = ("whisperx", "faster-whisper", "ctranslate2", "torch", "openai")


def _protocol_stream() -> TextIO:
    """Keep the real stdout for replies, and send everything else to stderr.

    WhisperX and other libraries print to stdout, which would corrupt the protocol.
    The descriptor is duplicated first, so even output from native code moves.

    Returns:
        A line-buffered text stream on the original stdout.
    """
    replies = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return replies


def _send(replies: TextIO, message: dict) -> None:
    """Write one JSON message on its own line."""
    replies.write(json.dumps(message, ensure_ascii=False) + "\n")
    replies.flush()


def _choose_whisper_model(model: str, compute_type: str) -> None:
    """Make Berta's WhisperX service load another Whisper model.

    `WhisperXTranscriptionService.__init__` hardcodes `large-v3` and `float32`. This
    replaces only those two values, before `app.config.ai` builds the service; its
    device, batch size, loading, and transcription stay Berta's.

    Args:
        model: A faster-whisper model name, such as `small`.
        compute_type: A CTranslate2 compute type, such as `float32` or `int8`.
    """
    from app.services import whisperx as berta_whisperx

    berta_config = berta_whisperx.WhisperXConfig

    class RequestedConfig(berta_config):
        """Berta's WhisperX configuration with the requested model."""

        def __init__(self, **values: object) -> None:
            """Override the model and compute type Berta passes."""
            values.update(model_version=model, compute_type=compute_type)
            super().__init__(**values)

    berta_whisperx.WhisperXConfig = RequestedConfig


def _note_instructions(folder: str, title: str) -> str:
    """Read a built-in note type the way Berta seeds its database.

    Args:
        folder: Berta's `BUILTIN_NOTETYPES_FOLDER`, with one folder per category.
        title: The note type's title, which is its file name without `.txt`.

    Returns:
        The instructions, stripped as Berta stores them.

    Raises:
        ValueError: If no category has that note type.
    """
    matches = sorted(Path(folder).glob(f"*/{title}.txt"))
    if not matches:
        raise ValueError(f"Berta has no built-in note type {title!r} in {folder}.")
    return matches[0].read_text(encoding="utf-8").strip()


class BertaTasks:
    """Berta's configured services, loaded once for the life of this process.

    Importing `app.config.ai` builds the transcription service, which loads the
    Whisper model, and the llama.cpp service, which lists the server's models.
    Both stay loaded until the process exits.
    """

    def __init__(self, *, note_model: str, note_type: str) -> None:
        """Import and configure Berta.

        Args:
            note_model: The model name llama-server serves, used for every note.
            note_type: Title of the built-in note type whose instructions to use.

        Raises:
            ValueError: If Whisper didn't load, the note model isn't served, or the
                note type is unknown.
        """
        from app.config import settings
        from app.config.ai import generative_ai_services, transcription_service
        from app.services.audio_processing import reformat_audio
        from app.tasks import generate_note, transcribe_audio

        # Berta's WhisperX service skips loading, silently, without whisperx.
        whisper = getattr(transcription_service, "config", None)
        if whisper is None:
            raise ValueError(
                "Berta's WhisperX service didn't load. Is whisperx installed?"
            )
        served = [
            model.name for service in generative_ai_services for model in service.models
        ]
        if note_model not in served:
            raise ValueError(f"Berta can't see model {note_model!r}; it sees {served}.")
        self.settings = settings
        self.note_model = note_model
        self.instructions = _note_instructions(
            settings.BUILTIN_NOTETYPES_FOLDER, note_type
        )
        self.transcription_service = transcription_service.service_name
        self.whisper_model = whisper.model_version
        self.whisper_compute_type = whisper.compute_type
        self._reformat_audio = reformat_audio
        self._transcribe_audio = transcribe_audio
        self._generate_note = generate_note

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a recording as Berta's web app does.

        The app converts every upload to MP3 at its default bitrate before saving
        it, then transcribes the saved MP3.

        Args:
            audio_path: The recording to transcribe.

        Returns:
            The transcript, as one string.
        """
        with open(audio_path, "rb") as original:
            converted, _ = self._reformat_audio(
                original, format="mp3", bitrate=self.settings.DEFAULT_AUDIO_BITRATE
            )
        converted.seek(0)
        filename = f"{Path(audio_path).parent.name}.mp3"
        output = asyncio.run(self._transcribe_audio(converted, filename, "audio/mpeg"))
        return output.transcript

    def write_note(self, transcript: str) -> dict:
        """Generate a note with Berta's prompt, note type, and llama.cpp adapter.

        Args:
            transcript: The consultation's transcript.

        Returns:
            The note text and the token counts Berta reports.
        """
        output = self._generate_note(
            self.note_model, self.instructions, NOTE_CONTEXT, transcript, OUTPUT_TYPE
        )
        return {
            "note": output.text,
            "prompt_tokens": output.promptTokens,
            "completion_tokens": output.completionTokens,
        }

    def describe(self) -> dict:
        """Report what was loaded, for the run's startup record."""
        packages = {}
        for name in _PACKAGES:
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
        return {
            "berta_version": self.settings.APP_VERSION,
            "transcription_service": self.transcription_service,
            "whisper_model": self.whisper_model,
            "whisper_compute_type": self.whisper_compute_type,
            "output_type": OUTPUT_TYPE,
            "python": sys.version.split()[0],
            "packages": packages,
        }


def _failure(error: BaseException) -> str:
    """Describe an exception in one line."""
    return f"{type(error).__name__}: {error}"


def handle(tasks: BertaTasks, request: dict) -> dict:
    """Run one request, returning a reply whether it succeeds or fails.

    Args:
        tasks: Berta's loaded services.
        request: `{"audio": path}` to transcribe and write a note, or
            `{"transcript": text}` to write a note only.

    Returns:
        The status, what was produced, the error if any, and each phase's seconds.
    """
    reply = {"status": "success", "error": None}
    transcript = request.get("transcript")
    # A failed sample is a result, not a reason to stop: report any error Berta or
    # its libraries raise, keeping what was produced before it.
    try:
        if transcript is None:
            started = time.perf_counter()
            transcript = tasks.transcribe(request["audio"])
            reply["transcription_seconds"] = time.perf_counter() - started
            reply["transcript"] = transcript
        started = time.perf_counter()
        reply.update(tasks.write_note(transcript))
        reply["note_seconds"] = time.perf_counter() - started
    except Exception as error:
        reply.update(status="error", error=_failure(error))
    return reply


def main() -> int:
    """Load Berta, then answer requests until stdin closes."""
    replies = _protocol_stream()
    started = time.perf_counter()
    # Any startup failure ends the worker; report it so the parent can show it.
    try:
        _choose_whisper_model(
            os.environ[WHISPER_MODEL_VARIABLE],
            os.environ[WHISPER_COMPUTE_TYPE_VARIABLE],
        )
        tasks = BertaTasks(
            note_model=os.environ[NOTE_MODEL_VARIABLE],
            note_type=os.environ[NOTE_TYPE_VARIABLE],
        )
    except Exception as error:
        _send(replies, {"status": "failed", "error": _failure(error)})
        return 1
    ready = {"status": "ready", "load_seconds": time.perf_counter() - started}
    _send(replies, ready | tasks.describe())
    for line in sys.stdin:
        _send(replies, handle(tasks, json.loads(line)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
