"""Convert TEME-v1 dialogues. For now, only the one used as a smoke test."""

import json
import shutil
import wave
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import DiarizedTranscript, Note, Segment
from .prepared import sample_directory, write_dataset, write_sample

_FIXTURE_NOTE = Path(__file__).parent / "assets" / "smoke_note_es.md"
_DEFAULT_SOURCE = Path("../scribnic/data/examples")
_SAMPLE_ID = "consulta_taquicardia_01"


def dialogue_transcript(dialogue: dict) -> DiarizedTranscript:
    """Read speaker turns. Duration estimates are not reliable, so times stay unset."""
    return DiarizedTranscript(
        tuple(Segment(turn["text"], turn["speaker"]) for turn in dialogue["turns"])
    )


def prepare_smoke(
    output: Path,
    *,
    source: Path | None = None,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> None:
    """Prepare one TEME-v1 dialogue with a handwritten fixture note.

    Args:
        output: Prepared dataset directory.
        source: Directory with `original.json` and `dialogue8.wav`.
        ids: Not supported; the smoke dataset has one sample.
        limit: Not supported; the smoke dataset has one sample.

    Raises:
        ValueError: If `ids` or `limit` are given.
    """
    if ids or limit:
        raise ValueError("The smoke dataset has a single sample.")
    source = source or _DEFAULT_SOURCE
    dialogue = json.loads((source / "original.json").read_text(encoding="utf-8"))
    directory = sample_directory(output, _SAMPLE_ID)
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "dialogue8.wav", directory / "audio.wav")
    with wave.open(str(directory / "audio.wav"), "rb") as audio:
        audio_seconds = audio.getnframes() / audio.getframerate()

    note = Note(_FIXTURE_NOTE.read_text(encoding="utf-8").strip())
    write_sample(
        output,
        _SAMPLE_ID,
        [dialogue_transcript(dialogue), note],
        {
            "audio_seconds": audio_seconds,
            "title": dialogue["title"],
            "context": dialogue["context"],
            "fixture_note": True,
        },
    )
    write_dataset(
        output,
        name="smoke",
        version="1",
        language="es",
        license="CC-BY-4.0",
        source="https://zenodo.org/records/17280661",
    )
