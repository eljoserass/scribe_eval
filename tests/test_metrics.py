"""Check metric values and, above all, how they aggregate."""

import unittest

from scribe_eval.artifacts import DiarizedTranscript, Note, Segment, Transcript
from scribe_eval.metrics import METRICS, resolve_metrics
from scribe_eval.metrics.note_format import section_positions
from scribe_eval.metrics.text import normalize_text
from scribe_eval.metrics.transcription import keep_terms
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
        self.assertEqual(names[:3], ["wer", "cer", "medical_term_wer"])
        self.assertEqual(names.count("wer"), 1)

    def test_unknown_names_fail(self):
        with self.assertRaises(ValueError):
            resolve_metrics(["nope"])


class MedicalTermTests(unittest.TestCase):
    def score(self, reference: str, predicted: str, language: str = "en"):
        """Compare two transcripts."""
        return METRICS["medical_term_wer"].score(
            prediction(transcript=Transcript.from_text(predicted)),
            Sample(
                "s",
                "all",
                language,
                {"transcript": Transcript.from_text(reference)},
            ),
        )

    def test_ordinary_words_do_not_count(self):
        score = self.score("you have asthma today", "we have asthma now")
        self.assertEqual(score.value, 0.0)

    def test_a_wrong_term_counts(self):
        score = self.score("you have asthma", "you have as ma")
        self.assertEqual(score.details["errors"], 1)

    def test_phrases_are_one_term(self):
        terms = frozenset({"insulin glargine", "insulin"})
        self.assertEqual(
            keep_terms("take insulin glargine daily", terms, 2), ["insulin_glargine"]
        )

    def test_an_invented_term_counts_without_a_reference_term(self):
        score = self.score("all is well today", "you have asthma")
        self.assertEqual(score.details["errors"], 1)

    def test_languages_without_a_list_do_not_apply(self):
        self.assertIsNone(self.score("tiene asma", "tiene asma", "es"))


class LevenshteinTests(unittest.TestCase):
    def test_counts_character_edits(self):
        score = METRICS["levenshtein"].score(
            prediction(note=Note("gato")), sample(note=Note("pato"))
        )
        self.assertEqual(score.value, 1)
        result = METRICS["levenshtein"].aggregate([score])
        self.assertEqual(result["per_character"], 0.25)


class OverlapTests(unittest.TestCase):
    def score(self, name: str, generated: str, reference: str, language: str = "en"):
        """Score a note against a reference."""
        case = Sample("s", "all", language, {"note": Note(reference)})
        return METRICS[name].score(prediction(note=Note(generated)), case)

    def test_identical_notes_are_perfect(self):
        text = "no chest pain and no fever since monday"
        self.assertEqual(self.score("rouge_1", text, text).value, 1.0)
        self.assertEqual(self.score("rouge_l", text, text).value, 1.0)
        self.assertAlmostEqual(self.score("bleu", text, text).value, 100.0)

    def test_rouge_keeps_accents_outside_english(self):
        self.assertEqual(self.score("rouge_1", "dolor", "calor", "es").value, 0.0)
        self.assertEqual(self.score("rouge_1", "día", "día", "es").value, 1.0)

    def test_corpus_bleu_is_not_the_mean_of_samples(self):
        metric = METRICS["bleu"]
        scores = [
            self.score(
                "bleu", "the patient has a mild fever", "the patient has a mild fever"
            ),
            self.score("bleu", "dose", "the patient has a mild fever today"),
        ]
        result = metric.aggregate(scores)
        self.assertNotEqual(result["corpus"], result["mean"])

    def test_meteor_only_applies_to_english(self):
        self.assertIsNone(self.score("meteor", "fiebre", "fiebre", "es"))

    def test_transcript_overlap_reads_the_transcripts(self):
        text = Transcript.from_text("no fever at all")
        score = METRICS["transcript_rouge_1"].score(
            prediction(transcript=text), sample(transcript=text)
        )
        self.assertEqual(score.value, 1.0)


class NoteQualityTests(unittest.TestCase):
    def test_compression_compares_note_and_conversation(self):
        metric = METRICS["compression_ratio"]
        score = metric.score(
            prediction(note=Note("no fever")),
            sample(transcript=Transcript.from_text("I have had no fever today")),
        )
        self.assertAlmostEqual(score.value, 2 / 6)
        self.assertAlmostEqual(metric.aggregate([score])["global"], 2 / 6)

    def test_simpler_text_reads_easier(self):
        metric = METRICS["flesch_reading_ease"]
        easy = metric.score(prediction(note=Note("The cat sat. It ran.")), sample())
        hard = metric.score(
            prediction(
                note=Note(
                    "Gastroenteritis complicated by dehydration necessitates rehydration."
                )
            ),
            sample(),
        )
        self.assertGreater(easy.value, hard.value)


class DiarizationTests(unittest.TestCase):
    def score(self, name: str, predicted: list[Segment], reference: list[Segment]):
        """Score predicted speaker turns against reference turns."""
        return METRICS[name].score(
            prediction(diarized_transcript=DiarizedTranscript(tuple(predicted))),
            sample(diarized_transcript=DiarizedTranscript(tuple(reference))),
        )

    def turns(self, *speakers: str) -> list[Segment]:
        """Build ten-second turns."""
        return [
            Segment("x", speaker, index * 10.0, (index + 1) * 10.0)
            for index, speaker in enumerate(speakers)
        ]

    def test_names_do_not_matter_only_the_partition(self):
        reference = self.turns("doctor", "patient")
        self.assertEqual(self.score("der", self.turns("A", "B"), reference).value, 0.0)
        self.assertEqual(self.score("jer", self.turns("A", "B"), reference).value, 0.0)

    def test_wrong_speaker_time_counts_as_confusion(self):
        reference = self.turns("doctor", "patient", "doctor", "patient")
        score = self.score("der", self.turns("A", "B", "A", "A"), reference)
        self.assertEqual(score.value, 0.25)
        self.assertEqual(score.details["confusion"], 10)

    def test_no_times_means_no_score(self):
        reference = [Segment("x", "doctor")]
        self.assertIsNone(self.score("der", self.turns("A"), reference))
