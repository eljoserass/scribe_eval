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

Each run lands in `runs/`, with a `report.html`. To continue or rescore one:

```bash
uv run scribe-eval resume runs/<run>
uv run scribe-eval resume runs/<run> --retry-errors
uv run scribe-eval score runs/<run>
```

`experiments/primock57_full.toml` runs all 57 consultations. That is hours on CPU,
so it is meant for a server: prepare the data there, or copy `data/` and set
`SCRIBE_EVAL_DATA`.

## Docs

- [Architecture](docs/architecture.md)
- [Adding a pipeline](docs/pipelines.md)
- [Datasets](docs/datasets.md)
- [Metrics](docs/metrics.md)

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
