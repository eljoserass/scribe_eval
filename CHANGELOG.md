# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). No release has been
tagged yet.

## [Unreleased]

### Added

- Evaluation core: one pipeline over one or more prepared datasets, resumable runs
  with frozen sample IDs, failures recorded as rows, and scoring kept separate from
  inference. `--no-score` saves predictions to score later.
- `Pipeline` interface and telemetry contract, so the harness knows no particular
  scribe. Scribnic is the first implementation, behind the `scribnic` extra.
- Dataset notes, including MEDIQA-Chat 2023's official scorer for ACI-BENCH.
- Datasets: Smoke, PriMock57 (English), and TEME-v1 (Spanish, TTS audio), converted
  to a common prepared format.
- Transcription metrics: WER and CER with Whisper's normalizers, ROUGE, and BLEU.
- Diarization metrics: DER and JER, behind the `diarization` extra.
- Note metrics: length, readability, compression, ROUGE, BLEU, METEOR, and
  Levenshtein against reference notes.
- Model metrics: BERTScore, BLEURT, and Clinical BERTScore, behind the `semantic`
  extra.
- Judge metrics: LINK and CORRECT replicated from ADS_evaluation, and `grounding`
  against the reference transcript. The judges are not validated.
- Resource metrics: time, real-time factor per phase, memory across the run, CPU,
  and token throughput, with hardware recorded per run.
- HTML report per run, `scribe-eval compare` for finished runs, and
  `scribe-eval runs` to list running, failed, and stale runs.

[Unreleased]: https://github.com/eljoserass/scribe_eval/commits/main
