# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html); 0.x minor versions may
break interfaces.

## [Unreleased]

### Tested on

- Linux, WSL2 x86_64: Berta AI Scribe on the 2-consultation PriMock57 smoke run.
- macOS 15.6, Apple M3 Ultra: Berta AI Scribe on all 57 PriMock57 consultations, no
  failed rows.

### Added

- Berta AI Scribe pipeline, `type = "berta"`: Berta's own transcription and note
  tasks, run in its own environment by a worker process, with Whisper small and the
  note model and llama-server Scribnic's English workflow uses. Telemetry covers
  both processes, phases, and tokens.
- `scripts/setup_berta.sh` installs Berta's environment and links the models on
  Linux and macOS, and `experiments/primock57/berta_smoke.toml` and
  `berta_full.toml` run PriMock57 with Scribnic's metrics.

### Changed

- ROUGE keeps precision and recall beside F1, per sample and averaged. On
  PriMock57, ROUGE-L precision tracked incorrect statements and recall tracked
  omissions better than F1 (Moramarco et al. 2022).
- Experiments are grouped by dataset and named after their pipeline, as in
  `experiments/primock57/scribnic_full.toml`. Each says whether it runs a scribe out of
  the box or a hybrid.

## [0.1.0] - 2026-11-19

First release: Scribnic runs PriMock57 end to end.

### Tested on

- macOS 15.6, Apple M3 Ultra: Scribnic `whisper_small_bonsai_en` on all 57 PriMock57
  consultations, 53 ok and 4 failed in Scribnic's note step, scored with the
  transcription, note format, note reference, and resource metrics.
- Linux, WSL2 x86_64: the same workflow on a 2-consultation smoke run.
- CI: Ubuntu and macOS, Python 3.12 and 3.15.

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
- Model metrics: BERTScore, BLEURT, NegBLEURT, and Clinical BERTScore, behind the
  `semantic` extra.
- Metrics can declare the languages they support; others show as n/a with a reason.
- Judge metrics: LINK and CORRECT replicated from ADS_evaluation, and `grounding`
  against the reference transcript. The judges are not validated.
- Resource metrics: time, real-time factor per phase, memory across the run, CPU,
  and token throughput, with hardware recorded per run.
- CI on Linux and macOS with Python 3.12 and 3.15, and Dependabot for GitHub
  Actions.
- `scripts/setup_scribnic.sh` installs Scribnic's English workflow and prepares
  PriMock57 on Linux and macOS.
- HTML report per run, `scribe-eval compare` for finished runs, and
  `scribe-eval runs` to list running, failed, and stale runs.

[Unreleased]: https://github.com/eljoserass/scribe_eval/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/eljoserass/scribe_eval/releases/tag/v0.1.0
