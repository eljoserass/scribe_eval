# scribe_eval

An evaluation harness for clinical scribes. It runs a scribe over datasets, scores
what comes out, and keeps every run resumable. Any scribe can plug in by implementing
one interface; [Scribnic](../scribnic) is the first.

A digital scribe listens to a consultation and drafts a note, usually SOAP, that a
clinician checks and files. It looks like a meeting summarizer, with three
differences:

- **Privacy.** Conversations and records are sensitive, and each hospital has its
  own rules. Running locally matters.
- **Errors cost more.** Saying a patient has a symptom, or doesn't, is not a typo.
- **Integration.** A block of text is not enough; it has to fit existing hospital
  systems.

## What gets evaluated

| Stage | Question |
| --- | --- |
| Audio capture | What audio can it process, and what breaks it? |
| Transcription | Audio to text. |
| Diarization | Who said what, and whether it helps: bad diarization may be worse than none. |
| Note | Does it use only what was said, and add nothing? With or without a reference note. |
| Resources | Time, memory, tokens. |

Pipelines differ. Some go straight from audio to note, so only those two artifacts
can be evaluated; others expose every step. Metrics declare what they need, and the
harness works out what applies.

Clinician workflow and IT integration are out of scope for now.

## Usage

Needs [uv](https://docs.astral.sh/uv/). The Scribnic pipeline also needs a Scribnic
checkout next to this repository.

```bash
uv sync
uv run scribe-eval prepare smoke
uv run scribe-eval run experiments/smoke.toml

uv run scribe-eval prepare primock57 --limit 3
uv run scribe-eval run experiments/primock57_sample.toml
```

TEME-v1 is the Spanish dataset: 90 consultations with TTS audio and no reference notes.
Preparing it needs `ffmpeg`, and its experiments need a local judge for `grounding`,
as described in [metrics](docs/metrics.md#link-and-correct).

```bash
uv run scribe-eval prepare teme
uv run scribe-eval run experiments/teme_es.toml
```

Heavy metrics are optional extras: `uv sync --extra semantic` for BERTScore, BLEURT,
and Clinical BERTScore (Torch), `--extra diarization` for DER and JER. See
[metrics](docs/metrics.md#model-metrics).

The PriMock57 experiments use an English Scribnic workflow in `workflows/`: Whisper
small with `language = "en"`, and an English SOAP prompt. Pipeline configuration isn't
part of the harness, so it lives on the `scribnic-workflows` branch, not on main.

Each run lands in `runs/`, with a `report.html`. `--no-score` runs only the pipeline
and saves its predictions, so a long run needs no judge or metric models; `score`
finishes it later. To continue or rescore one:

```bash
uv run scribe-eval run experiments/primock57_full.toml --no-score
uv run scribe-eval resume runs/<run>
uv run scribe-eval resume runs/<run> --retry-errors
uv run scribe-eval score runs/<run>
```

To see what is running, what failed and why, or every run, with progress and times:

```bash
uv run scribe-eval runs
uv run scribe-eval runs failed
uv run scribe-eval runs all
```

A run whose process died shows as `stale`. Each run also keeps its predictions,
artifacts, telemetry, and status in its directory.

To compare pipelines or workflows, compare their runs. The table puts coverage and
failures first, and warns when runs scored different samples or metric versions:

```bash
uv run scribe-eval compare runs/<run-a> runs/<run-b>
```

`experiments/primock57_full.toml` runs all 57 consultations. That is hours on CPU,
so it is meant for a server: prepare the data there, or copy `data/` and set
`SCRIBE_EVAL_DATA`.

## Docs

- [Architecture](docs/architecture.md)
- [Adding a pipeline](docs/pipelines.md)
- [Datasets](docs/datasets.md)
- [Metrics](docs/metrics.md)
- [Evaluation notes](docs/evaluation_notes.md)
- [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), and
  [Changelog](CHANGELOG.md)

## License

MIT; see [LICENSE](LICENSE). Vendored files keep their own notices:
Whisper's normalizers in
[`metrics/whisper_normalizers`](src/scribe_eval/metrics/whisper_normalizers/LICENSE)
(MIT, OpenAI), and MedWER's medical term list, described in
[`metrics/assets/NOTICE`](src/scribe_eval/metrics/assets/NOTICE). The LINK and
CORRECT prompts are fetched at first use and not redistributed, since their
repository has no license.

## References

- [Ambient scribe evaluation datasets](https://huggingface.co/collections/sarahgebauer/ambient-scribe-evaluation-datasets)
- [Afrispeech-Dialog, NAACL 2025](https://aclanthology.org/2025.naacl-long.426.pdf)
- [MedSynth](https://arxiv.org/html/2508.01401v2)
- [Improving clinical note generation (CliniKnote)](https://arxiv.org/html/2408.14568v1)
- [SOAP notes, StatPearls](https://www.ncbi.nlm.nih.gov/books/NBK482263/)
- [An evaluation framework for ambient digital scribing tools](https://www.nature.com/articles/s41746-025-01622-1)
- [ADS_evaluation](https://github.com/AutoEvaluation/ADS_evaluation)
- [EvalLLM4Health](https://github.com/AutoEvaluation/EvalLLM4Health)
- [AU-Harness](https://github.com/ServiceNow/AU-Harness)
