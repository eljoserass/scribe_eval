"""Convert PriMock57: mock primary care consultations with audio and notes."""

import hashlib
import json
import re
import urllib.request
import wave
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ..artifacts import DiarizedTranscript, Note, Segment
from .prepared import sample_directory, write_dataset, write_sample

REPOSITORY = "babylonhealth/primock57"
COMMIT = "cd2ac707ad03cb4d2531f4ec6b90c659bf4357c5"
_RAW_URL = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}"
_MEDIA_URL = f"https://media.githubusercontent.com/media/{REPOSITORY}/{COMMIT}"
_TREE_URL = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1"
_SPEAKERS = ("doctor", "patient")
_SAMPLE_RATE = 16000

_INTERVAL = re.compile(
    r'xmin = ([\d.]+)\s*\n\s*xmax = ([\d.]+)\s*\n\s*text = "(.*?)"\s*\n', re.DOTALL
)
_MARKUP = re.compile(r"</?UNSURE>|<UNIN/>|<INAUDIBLE_SPEECH/>")
_NOTE_FILE = re.compile(r"notes/(day\d+_consultation\d+)\.json")


def _fetch(url: str) -> bytes:
    """Download a small file."""
    request = urllib.request.Request(url, headers={"User-Agent": "scribe-eval"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _available_ids(source: Path | None) -> list[str]:
    """List consultations in a local copy, or in the pinned repository."""
    if source is not None:
        paths = [f"notes/{path.name}" for path in (source / "notes").glob("*.json")]
    else:
        tree = json.loads(_fetch(_TREE_URL))["tree"]
        paths = [entry["path"] for entry in tree]
    return sorted(m[1] for path in paths if (m := _NOTE_FILE.fullmatch(path)))


def _download_file(relative: str, source: Path) -> None:
    """Download a text file unless it is already there."""
    target = source / relative
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_fetch(f"{_RAW_URL}/{relative}"))


def _download_audio(relative: str, source: Path) -> None:
    """Download a Git LFS file directly, verifying it against its pointer.

    This avoids requiring git-lfs on the machine that prepares the data.
    """
    target = source / relative
    if target.exists():
        return
    pointer = _fetch(f"{_RAW_URL}/{relative}").decode("ascii")
    expected = re.search(r"oid sha256:([0-9a-f]{64})", pointer)
    if expected is None:
        raise ValueError(f"Expected a Git LFS pointer for {relative}.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    request = urllib.request.Request(
        f"{_MEDIA_URL}/{relative}", headers={"User-Agent": "scribe-eval"}
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=120) as response:
        with temporary.open("wb") as stream:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                stream.write(chunk)
    if digest.hexdigest() != expected[1]:
        temporary.unlink()
        raise ValueError(f"Checksum mismatch for {relative}.")
    temporary.replace(target)


def _download_consultation(sample_id: str, source: Path) -> None:
    """Fetch the raw files of one consultation."""
    _download_file(f"notes/{sample_id}.json", source)
    for speaker in _SPEAKERS:
        _download_file(f"transcripts/{sample_id}_{speaker}.TextGrid", source)
        _download_audio(f"audio/{sample_id}_{speaker}.wav", source)


def parse_textgrid(text: str, speaker: str) -> list[Segment]:
    """Read one speaker's utterances, removing transcriber markup.

    `<UNSURE>` keeps its words. `<UNIN/>` and `<INAUDIBLE_SPEECH/>` mark audio nobody
    could transcribe.
    """
    segments = []
    for start, end, content in _INTERVAL.findall(text):
        cleaned = " ".join(_MARKUP.sub(" ", content.replace('""', '"')).split())
        if cleaned:
            segments.append(
                Segment(cleaned, speaker, float(start), float(end), speaker)
            )
    return segments


def _read_channel(path: Path) -> np.ndarray:
    """Read one speaker's audio, which must already be mono PCM16 at 16 kHz."""
    with wave.open(str(path), "rb") as audio:
        params = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
        if params != (1, 2, _SAMPLE_RATE):
            raise ValueError(f"Unexpected audio format {params} in {path}.")
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")


def mix_channels(doctor: Path, patient: Path, output: Path) -> float:
    """Average both speakers into one mono WAV, as `sox -m` does.

    Returns:
        Duration of the mix in seconds.
    """
    first = _read_channel(doctor)
    second = _read_channel(patient)
    length = max(len(first), len(second))
    mixed = np.zeros(length, dtype=np.int32)
    mixed[: len(first)] += first
    mixed[: len(second)] += second
    with wave.open(str(output), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(_SAMPLE_RATE)
        audio.writeframes((mixed // 2).astype("<i2").tobytes())
    return length / _SAMPLE_RATE


def _convert_consultation(sample_id: str, source: Path, output: Path) -> None:
    """Write one consultation in the common format."""
    segments = []
    for speaker in _SPEAKERS:
        path = source / "transcripts" / f"{sample_id}_{speaker}.TextGrid"
        segments += parse_textgrid(path.read_text(encoding="utf-8"), speaker)
    segments.sort(key=lambda segment: segment.start)

    raw_note = json.loads((source / "notes" / f"{sample_id}.json").read_text("utf-8"))
    directory = sample_directory(output, sample_id)
    directory.mkdir(parents=True, exist_ok=True)
    audio_seconds = mix_channels(
        source / "audio" / f"{sample_id}_doctor.wav",
        source / "audio" / f"{sample_id}_patient.wav",
        directory / "audio.wav",
    )
    write_sample(
        output,
        sample_id,
        [DiarizedTranscript(tuple(segments)), Note(raw_note["note"])],
        {
            "audio_seconds": audio_seconds,
            "presenting_complaint": raw_note["presenting_complaint"],
            "highlights": raw_note["highlights"],
        },
    )


def prepare(
    output: Path,
    *,
    source: Path | None = None,
    ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> None:
    """Download what is missing and convert consultations.

    Args:
        output: Prepared dataset directory.
        source: Raw files, in the repository's layout. Defaults to `<output>/raw`,
            where missing files are downloaded from a pinned commit.
        ids: Consultations to convert; all by default.
        limit: Keep only the first consultations.

    Raises:
        ValueError: If a requested ID does not exist.
    """
    download = source is None
    available = _available_ids(source)
    source = source or output / "raw"
    unknown = sorted(set(ids or ()) - set(available))
    if unknown:
        raise ValueError(f"Unknown consultations: {', '.join(unknown)}")
    selected = sorted(ids) if ids else available
    if limit is not None:
        selected = selected[:limit]

    for sample_id in selected:
        if download:
            _download_consultation(sample_id, source)
        _convert_consultation(sample_id, source, output)
    write_dataset(
        output,
        name="primock57",
        version=COMMIT,
        language="en",
        license="CC-BY-4.0",
        source=f"https://github.com/{REPOSITORY}",
    )
