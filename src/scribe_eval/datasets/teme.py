"""Convert TEME-v1: Spanish consultations with TTS audio and no notes.

Villarroya-Martinez et al., "TEME: A Multi-Agent Evaluation Framework for Spanish
Medical Speech Recognition", AAMAS 2026. https://doi.org/10.65109/RHNT7438
Per the paper, 23 dialogues are real anonymized consultations and 67 are synthetic;
the archive doesn't mark which.

`prepare` converts the whole dataset from its Zenodo ZIP. `prepare_smoke` converts the
one dialogue used to test the plumbing, with a handwritten fixture note.
"""

import hashlib
import json
import re
import shutil
import subprocess
import urllib.request
import wave
import zipfile
from collections.abc import Sequence
from pathlib import Path

from ..artifacts import DiarizedTranscript, Note, Segment
from .prepared import sample_directory, write_dataset, write_sample

RECORD = "17280661"
SOURCE_URL = f"https://zenodo.org/records/{RECORD}"
_ARCHIVE = "TEME-medical-dataset.zip"
_ARCHIVE_URL = f"https://zenodo.org/api/records/{RECORD}/files/{_ARCHIVE}/content"
_ARCHIVE_MD5 = "cbb9f0cecb1999d662f05aac0aa048be"
_DIALOGUE_MEMBER = re.compile(r"dialogues/(dialogue(\d+))_original\.json")
_SAMPLE_RATE = 16000

_FIXTURE_NOTE = Path(__file__).parent / "assets" / "smoke_note_es.md"
_DEFAULT_SOURCE = Path("../scribnic/data/examples")
_SAMPLE_ID = "consulta_taquicardia_01"


def dialogue_transcript(dialogue: dict) -> DiarizedTranscript:
    """Read speaker turns. Duration estimates are not reliable, so times stay unset."""
    return DiarizedTranscript(
        tuple(Segment(turn["text"], turn["speaker"]) for turn in dialogue["turns"])
    )


def _audio_seconds(path: Path) -> float:
    """Read the duration of a WAV file."""
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def prepare(
    output: Path,
    *,
    source: Path | None = None,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> None:
    """Convert TEME-v1 dialogues, downloading the archive if needed.

    Each `dialogueN` becomes a sample with 16 kHz audio and its diarized script. The
    `context` summary goes to metadata only; it is not a reference note.

    Args:
        output: Prepared dataset directory.
        source: The Zenodo ZIP. Defaults to `<output>/raw/TEME-medical-dataset.zip`,
            downloaded and checked against its MD5 when missing.
        ids: Dialogues to convert, such as `dialogue12`; all by default.
        limit: Keep only the first dialogues, in numeric order.

    Raises:
        ValueError: If a requested ID does not exist, the download is corrupt, or
            ffmpeg cannot decode an MP3.
        FileNotFoundError: If ffmpeg is not installed.
    """
    archive = source or output / "raw" / _ARCHIVE
    if source is None:
        _download_archive(archive)
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError("TEME-v1 audio is MP3; converting it needs ffmpeg.")

    with zipfile.ZipFile(archive) as bundle:
        selected = _select(_dialogue_ids(bundle), ids, limit)
        for sample_id in selected:
            _convert_dialogue(bundle, sample_id, output)
    write_dataset(
        output,
        name="teme",
        version="1",
        language="es",
        license="CC-BY-4.0",
        source=SOURCE_URL,
    )


def _download_archive(target: Path) -> None:
    """Download the Zenodo ZIP unless it is already there, verifying its MD5."""
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    request = urllib.request.Request(
        _ARCHIVE_URL, headers={"User-Agent": "scribe-eval"}
    )
    digest = hashlib.md5()
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as stream:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                stream.write(chunk)
    if digest.hexdigest() != _ARCHIVE_MD5:
        temporary.unlink()
        raise ValueError(f"Checksum mismatch for {_ARCHIVE}.")
    temporary.replace(target)


def _dialogue_ids(bundle: zipfile.ZipFile) -> list[str]:
    """List dialogue IDs in the archive, in numeric order."""
    matches = [_DIALOGUE_MEMBER.fullmatch(name) for name in bundle.namelist()]
    found = [match for match in matches if match]
    return [match[1] for match in sorted(found, key=lambda match: int(match[2]))]


def _select(
    available: list[str], ids: Sequence[str] | None, limit: int | None
) -> list[str]:
    """Apply `ids` and `limit`, keeping the archive's numeric order.

    Raises:
        ValueError: If a requested ID is not in the archive.
    """
    unknown = sorted(set(ids or ()) - set(available))
    if unknown:
        raise ValueError(f"Unknown dialogues: {', '.join(unknown)}")
    selected = [sample_id for sample_id in available if not ids or sample_id in ids]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _convert_dialogue(bundle: zipfile.ZipFile, sample_id: str, output: Path) -> None:
    """Write one dialogue in the common format."""
    dialogue = json.loads(bundle.read(f"dialogues/{sample_id}_original.json"))
    directory = sample_directory(output, sample_id)
    directory.mkdir(parents=True, exist_ok=True)
    mp3 = bundle.read(f"audio_dialogues/{sample_id}.modify.mp3")
    decode_mp3(mp3, directory / "audio.wav")
    write_sample(
        output,
        sample_id,
        [dialogue_transcript(dialogue)],
        {
            "audio_seconds": _audio_seconds(directory / "audio.wav"),
            "scenario_id": dialogue["scenario_id"],
            "title": dialogue["title"],
            "context": dialogue["context"],
            "synthetic_audio": "tts",
        },
    )


def decode_mp3(data: bytes, output: Path) -> None:
    """Convert MP3 bytes to mono PCM16 WAV at 16 kHz with ffmpeg.

    TEME-v1 audio is 8 kHz, so resampling adds no bandwidth; it only matches the
    format every pipeline accepts.

    Args:
        data: The MP3 file's contents.
        output: WAV file to write, replacing any existing one.

    Raises:
        ValueError: If ffmpeg cannot decode the data.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-ac",
        "1",
        "-ar",
        str(_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        "-y",
        str(output),
    ]
    try:
        subprocess.run(command, input=data, capture_output=True, check=True)
    except subprocess.CalledProcessError as error:
        message = error.stderr.decode(errors="replace").strip()
        raise ValueError(
            f"ffmpeg could not decode {output.parent.name}: {message}"
        ) from error


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

    note = Note(_FIXTURE_NOTE.read_text(encoding="utf-8").strip())
    write_sample(
        output,
        _SAMPLE_ID,
        [dialogue_transcript(dialogue), note],
        {
            "audio_seconds": _audio_seconds(directory / "audio.wav"),
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
        source=SOURCE_URL,
    )
