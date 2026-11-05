"""Check conversion from raw formats to the common one."""

import json
import tempfile
import unittest
import wave
from pathlib import Path

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
