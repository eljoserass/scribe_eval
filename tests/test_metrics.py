"""Check metric values and, above all, how they aggregate."""

import unittest

from scribe_eval.artifacts import Note, Transcript
from scribe_eval.metrics import METRICS, resolve_metrics
from scribe_eval.metrics.note_format import section_positions
from scribe_eval.metrics.text import normalize_text
from scribe_eval.telemetry import ModelUsage, Phase, ProcessUsage, SampleTelemetry
from scribe_eval.types import Prediction, Sample, Score


def prediction(input_kind="audio", telemetry=None, **artifacts) -> Prediction:
    """Build a successful prediction."""
    telemetry = telemetry or SampleTelemetry(wall_seconds=2.0)
    return Prediction(
        "d",
        "s",
        "clean",
        0,
        input_kind,
        "success",
        None,
        artifacts,
        telemetry,
        "x",
        "t",
    )


def sample(**artifacts) -> Sample:
    """Build an English sample."""
    return Sample("s", "all", "en", artifacts, {"audio_seconds": 4.0})


class TextTests(unittest.TestCase):
    def test_normalization_spells_numbers_and_drops_punctuation(self):
        self.assertEqual(normalize_text("Take 3, Doctor!", "en"), "take three doctor")

    def test_normalization_drops_hesitations(self):
        self.assertEqual(normalize_text("Um, yeah, uh, okay", "en"), "yeah okay")

    def test_normalization_uses_the_language(self):
        self.assertEqual(normalize_text("Toma 3", "es"), "toma tres")


class TranscriptionTests(unittest.TestCase):
    def test_wer_counts_errors_against_the_reference(self):
        score = METRICS["wer"].score(
            prediction(transcript=Transcript.from_text("a b x")),
            sample(transcript=Transcript.from_text("a b c d")),
        )
        self.assertEqual(score.value, 0.5)

    def test_empty_prediction_is_a_full_error(self):
        score = METRICS["wer"].score(
            prediction(transcript=Transcript.from_text("")),
            sample(transcript=Transcript.from_text("a b")),
        )
        self.assertEqual(score.value, 1.0)
        self.assertEqual(score.details["reference_size"], 2)

    def test_global_wer_weighs_by_length_unlike_the_sample_mean(self):
        scores = [
            Score("wer", 1.0, {"errors": 1, "reference_size": 1}),
            Score("wer", 0.0, {"errors": 0, "reference_size": 99}),
        ]
        result = METRICS["wer"].aggregate(scores)
        self.assertEqual(result["global"], 0.01)
        self.assertEqual(result["sample_mean"], 0.5)


class NoteTests(unittest.TestCase):
    def soap(self, text: str) -> float:
        """Score a note's sections."""
        note = Note(text)
        return METRICS["soap_sections"].score(prediction(note=note), sample()).value

    def test_accepts_plain_and_markdown_labels(self):
        self.assertEqual(self.soap("S — a\nO — b\nA — c\nP — d"), 1.0)
        self.assertEqual(
            self.soap("**S — Subjective:** a\n**O:** b\n**A:** c\nP: d"), 1.0
        )

    def test_rejects_a_missing_section_a_title_and_wrong_order(self):
        self.assertEqual(self.soap("S — a\nO — b\nP — d"), 0.0)
        self.assertEqual(self.soap("SOAP note\nS — a\nO — b\nA — c\nP — d"), 0.0)
        self.assertIsNone(section_positions("O — b\nS — a\nA — c\nP — d"))

    def test_identical_notes_have_full_rouge(self):
        note = Note("no chest pain")
        score = METRICS["rouge_l"].score(prediction(note=note), sample(note=note))
        self.assertEqual(score.value, 1.0)


class ResourceTests(unittest.TestCase):
    def test_real_time_factor_only_applies_to_audio_input(self):
        metric = METRICS["real_time_factor"]
        self.assertEqual(metric.score(prediction(), sample()).value, 0.5)
        self.assertIsNone(metric.score(prediction("transcript"), sample()))

    def test_missing_token_usage_is_skipped(self):
        self.assertIsNone(METRICS["prompt_tokens"].score(prediction(), sample()))


class TelemetryMetricTests(unittest.TestCase):
    def rich_prediction(self) -> Prediction:
        """Build a prediction from a pipeline that reports everything."""
        telemetry = SampleTelemetry(
            wall_seconds=4.5,
            phases=(
                Phase("transcription", 3.0),
                Phase("note", 1.0),
                Phase("generation", 0.5, parent="note"),
            ),
            processes=(ProcessUsage("asr", 5, 4.0), ProcessUsage("llm", 9, 1.0)),
            models=(ModelUsage("note", 10, 5, 100.0, 20.5),),
        )
        return prediction(telemetry=telemetry)

    def test_phases_stay_separate(self):
        metric = METRICS["phase_seconds"]
        result = metric.aggregate([metric.score(self.rich_prediction(), sample())])
        self.assertEqual(result["transcription"], 3.0)
        self.assertEqual(result["generation"], 0.5)

    def test_phase_ratio_divides_total_time_by_total_audio(self):
        metric = METRICS["phase_real_time_factor"]
        score = metric.score(self.rich_prediction(), sample())
        self.assertEqual(metric.aggregate([score])["transcription"], 0.75)

    def test_outside_time_comes_from_the_harness(self):
        score = METRICS["elapsed_seconds"].score(self.rich_prediction(), sample())
        self.assertEqual(score.value, 4.5)

    def test_memory_is_reported_per_process(self):
        metric = METRICS["process_peak_rss_bytes"]
        score = metric.score(self.rich_prediction(), sample())
        self.assertEqual(score.value, 9)
        self.assertEqual(metric.aggregate([score])["asr"], 5)

    def test_speed_and_tokens_read_the_contract(self):
        speed = METRICS["generation_tokens_per_second"]
        self.assertEqual(speed.score(self.rich_prediction(), sample()).value, 20.5)
        self.assertEqual(
            METRICS["prompt_tokens"].score(self.rich_prediction(), sample()).value, 10
        )

    def test_a_pipeline_without_telemetry_only_loses_those_metrics(self):
        for name in ("phase_seconds", "process_cpu_cores", "prompt_tokens"):
            self.assertIsNone(METRICS[name].score(prediction(), sample()))
        self.assertIsNone(
            METRICS["generation_tokens_per_second"].score(prediction(), sample())
        )
        self.assertEqual(
            METRICS["elapsed_seconds"].score(prediction(), sample()).value, 2.0
        )


class RegistryTests(unittest.TestCase):
    def test_groups_expand_without_duplicates(self):
        names = [metric.name for metric in resolve_metrics(["wer", "transcription"])]
        self.assertEqual(names, ["wer", "cer"])

    def test_unknown_names_fail(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["nope"])
