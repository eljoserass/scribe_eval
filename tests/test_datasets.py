"""Check conversion from raw formats to the common one."""

import json
import shutil
import subprocess
import tempfile
import unittest
import wave
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from scribe_eval.datasets import primock57, teme
from scribe_eval.datasets.prepared import PreparedDataset

TEXTGRID = """
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = ""
        intervals [2]:
            xmin = 1.5
            xmax = 3.25
            text = "Hello <UNSURE>doctor</UNSURE> <UNIN/> <INAUDIBLE_SPEECH/> ""yes""."
        intervals [3]:
            xmin = 3.25
            xmax = 4
            text = "<UNIN/>"
"""


def write_wav(path: Path, samples: list[int]) -> None:
    """Write a short mono PCM16 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(np.array(samples, dtype="<i2").tobytes())


class PriMockTests(unittest.TestCase):
    def test_textgrid_keeps_unsure_words_and_drops_empty_intervals(self):
        segments = primock57.parse_textgrid(TEXTGRID, "doctor")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, 'Hello doctor "yes".')
        self.assertEqual((segments[0].start, segments[0].end), (1.5, 3.25))
        self.assertEqual(segments[0].role, "doctor")

    def test_mix_averages_channels_and_pads_the_shorter_one(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_wav(root / "d.wav", [100, 200, 300])
            write_wav(root / "p.wav", [300])
            duration = primock57.mix_channels(
                root / "d.wav", root / "p.wav", root / "m.wav"
            )
            with wave.open(str(root / "m.wav"), "rb") as audio:
                mixed = np.frombuffer(audio.readframes(3), dtype="<i2")
        self.assertEqual(mixed.tolist(), [200, 100, 150])
        self.assertEqual(duration, 3 / 16000)

    def test_local_copy_converts_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            sample = "day1_consultation01"
            note = {"note": "Imp: gastroenteritis", "presenting_complaint": "x"}
            note["highlights"] = []
            (raw / "notes").mkdir(parents=True)
            (raw / "transcripts").mkdir()
            (raw / "notes" / f"{sample}.json").write_text(json.dumps(note), "utf-8")
            for speaker in ("doctor", "patient"):
                path = raw / "transcripts" / f"{sample}_{speaker}.TextGrid"
                path.write_text(TEXTGRID, "utf-8")
                write_wav(raw / "audio" / f"{sample}_{speaker}.wav", [1, 2])

            output = Path(directory) / "primock57"
            primock57.prepare(output, source=raw)
            dataset = PreparedDataset(output)
            loaded = dataset.sample(sample)
            self.assertEqual(dataset.provides, {"audio", "diarized_transcript", "note"})
            self.assertEqual(loaded.language, "en")
            self.assertEqual(loaded.artifact("note").text, "Imp: gastroenteritis")
            self.assertEqual(len(loaded.artifact("transcript").segments), 2)


class TemeTests(unittest.TestCase):
    def test_turns_keep_speakers_but_not_estimated_times(self):
        dialogue = {
            "turns": [{"speaker": "Doctor", "text": "Hola", "duration_estimate": 3}]
        }
        segment = teme.dialogue_transcript(dialogue).segments[0]
        self.assertEqual(segment.speaker, "Doctor")
        self.assertIsNone(segment.start)


def write_teme_zip(path: Path, numbers: list[int]) -> None:
    """Write an archive in TEME-v1's layout, with placeholder MP3 bytes."""
    with zipfile.ZipFile(path, "w") as bundle:
        for number in numbers:
            dialogue = {
                "scenario_id": f"consulta_{number}",
                "title": "Consulta",
                "context": "Resumen que no es una nota",
                "turns": [
                    {"speaker": "Doctor", "text": "¿Qué le pasa?"},
                    {"speaker": "PacienteMujer", "text": "Me duele la cabeza."},
                ],
            }
            name = f"dialogues/dialogue{number}_original.json"
            bundle.writestr(name, json.dumps(dialogue))
            bundle.writestr(f"audio_dialogues/dialogue{number}.modify.mp3", b"mp3")


def fake_decode(data: bytes, output: Path) -> None:
    """Stand in for ffmpeg with half a second of silence."""
    write_wav(output, [0] * 8000)


class TemeArchiveTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.archive = self.root / "teme.zip"
        write_teme_zip(self.archive, [1, 2, 10])

    def prepare(self, **selection) -> PreparedDataset:
        """Convert the fake archive without ffmpeg."""
        output = self.root / "teme"
        with (
            mock.patch.object(teme, "decode_mp3", fake_decode),
            mock.patch.object(teme.shutil, "which", return_value="/usr/bin/ffmpeg"),
        ):
            teme.prepare(output, source=self.archive, **selection)
        return PreparedDataset(output)

    def test_converts_audio_and_script_without_a_note(self):
        dataset = self.prepare()
        sample = dataset.sample("dialogue2")
        self.assertEqual(dataset.provides, {"audio", "diarized_transcript"})
        self.assertEqual(dataset.language, "es")
        self.assertEqual(
            sample.artifact("diarized_transcript").segments[1].speaker, "PacienteMujer"
        )
        self.assertEqual(sample.metadata["audio_seconds"], 0.5)
        self.assertEqual(sample.metadata["scenario_id"], "consulta_2")

    def test_limit_follows_numeric_order(self):
        dataset = self.prepare(limit=2)
        self.assertEqual(dataset.splits["all"], ["dialogue1", "dialogue2"])

    def test_ids_select_dialogues(self):
        dataset = self.prepare(ids=["dialogue10"])
        self.assertEqual(dataset.splits["all"], ["dialogue10"])

    def test_unknown_id_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "dialogue99"):
            self.prepare(ids=["dialogue99"])

    def test_missing_ffmpeg_is_explained(self):
        with mock.patch.object(teme.shutil, "which", return_value=None):
            with self.assertRaisesRegex(FileNotFoundError, "ffmpeg"):
                teme.prepare(self.root / "teme", source=self.archive)

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_decode_writes_16_khz_mono_pcm(self):
        mp3 = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=8000:duration=0.5",
                "-f",
                "mp3",
                "pipe:1",
            ],
            capture_output=True,
            check=True,
        ).stdout
        output = self.root / "audio.wav"
        teme.decode_mp3(mp3, output)
        with wave.open(str(output), "rb") as audio:
            params = (audio.getnchannels(), audio.getsampwidth(), audio.getframerate())
        self.assertEqual(params, (1, 2, 16000))

    @unittest.skipUnless(shutil.which("ffmpeg"), "needs ffmpeg")
    def test_undecodable_audio_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "could not decode"):
            teme.decode_mp3(b"not audio", self.root / "audio.wav")
