# Architecture

`scribe_eval` connects data, a scribe, and metrics. It knows nothing about any
particular scribe: each one implements `Pipeline`, and the experiment picks which.

## Principles

1. **Pipelines plug in.** Only an implementation imports its scribe. See
   [Adding a pipeline](pipelines.md).
2. **One experiment, one pipeline.** It can cover several datasets. To compare
   pipelines, compare finished runs.
3. **Inference and scoring are separate.** A new metric or judge never regenerates
   notes.
4. **An experiment is a definition; a run is an execution.** `experiments/` holds
   versioned TOML. `runs/` holds what was generated, with sample IDs frozen.
5. **Datasets convert once to a common format.** One loader reads them all.
6. **Explicit registries.** Datasets and metrics live in plain dictionaries.
7. **The CLI is a frontend.** Everything works from `application.py`.

## Flow

```text
prepare    raw data → data/<dataset>/
validate   experiment + dataset + pipeline → errors or n/a, no models loaded
infer      sample → pipeline.run → predictions.jsonl
score      prediction + sample → scores.jsonl
aggregate  scores → results.json
report     results + predictions → report.html
```

## Experiments

```toml
name = "primock57-sample"

[pipeline]
type = "scribnic"      # the rest goes to the implementation
workflow = "../workflows/whisper_small_bonsai_en.toml"

[[datasets]]
name = "primock57"
split = "all"
input = "audio"        # what enters the pipeline
limit = 3              # or fraction = 0.2, or ids = [...]
seed = 42
metrics = ["transcription", "note_format", "resources"]  # metrics or groups
```

`input` matters: a dataset with audio and transcripts can feed a pipeline either
way, and that changes what can be measured. The TOML keeps the intent (`limit`).
`manifest.json` keeps the resolved IDs, so a resume uses the same samples.

## Data structures

`artifacts.py` defines `AudioFile`, `Transcript`, `DiarizedTranscript`, and `Note`,
with plain JSON forms. See `types.py` for the rest.

```python
class Sample:
    id: str
    split: str
    language: str
    artifacts: dict[ArtifactKind, Artifact]  # references from the dataset
    metadata: dict[str, object]

class Prediction:
    dataset: str
    sample_id: str
    variant: str          # "clean" until perturbations exist
    repeat: int           # which repetition of this input; always 0 for now
    input_kind: ArtifactKind
    status: Literal["success", "error"]
    error: str | None
    artifacts: dict[ArtifactKind, Artifact]  # only what the pipeline produced
    telemetry: SampleTelemetry
    run_directory: str

class Requires:
    predicted: tuple[ArtifactKind, ...]
    reference: tuple[ArtifactKind, ...]

class Score:
    metric: str
    value: float
    details: dict[str, float]  # e.g. {"errors": 12, "reference_size": 340}

class Metric(Protocol):
    name: str
    group: str
    version: int
    requires: Requires
    def score(self, prediction, sample) -> Score | None: ...
    def aggregate(self, scores) -> dict[str, float]: ...
```

- A failed run keeps what it produced. If the note is rejected, WER still counts.
- `aggregate` belongs to each metric: global WER is total errors over total words,
  not a mean of per-sample rates. That's why `Score` carries `details`.
- `diarized_transcript` also satisfies `transcript`, through `without_speakers()`.

## Validation

Runs before any model loads. What a pipeline produces comes from `produces()`.

| Case | Result |
| --- | --- |
| Dataset lacks the input or a required reference | Error |
| Pipeline doesn't accept the input | Error |
| Pipeline doesn't produce the artifact, like WER on audio-to-note | `n/a`, with the reason |
| A transcript goes in and WER is requested | `n/a`, with the reason |

## Isolation

Samples are independent by default. The harness reopens the pipeline for each one, so
no cache or memory carries over, at the cost of a model load per sample. Load time
isn't counted in a sample's timing.

`isolate_samples = false` keeps the pipeline open across samples. Use it to measure a
long-lived deployment, where carry-over is the point.

Carry-over is real. Over 10 notes from 4 transcripts on one `llama-server`, its prompt
cache took memory from 1.38 GB to 1.88 GB, and repeated prompts ran up to 4 times
faster. With the cache off, memory stayed at 1.38 GB and timings stayed honest.

## Runs

Laid out like [HELM](https://crfm-helm.readthedocs.io/en/v0.5.3/tutorial/): spec,
per-instance state, per-instance stats, and aggregates in separate files.

```text
runs/20261105-153000-primock57-sample-1a2b3c/
├── experiment.toml      # exact copy
├── manifest.json        # IDs, versions, hardware, effective pipeline
├── status.json          # running, completed, interrupted, or failed
├── startup.jsonl        # model load times
├── results.json         # dataset × metric, overall, telemetry summary
├── report.html          # self-contained; includes notes, so treat it like the data
├── telemetry/execution-1/  # OTLP export, one folder per pipeline open
└── datasets/primock57/
    ├── predictions.jsonl
    ├── scores.jsonl
    └── samples/<sample_id>/attempt-1/
        ├── note.json    # what the pipeline produced
        └── pipeline/    # anything else it wants to keep
```

- **JSONL is the source of truth.** One row per sample, flushed as it finishes.
- **Resume** skips recorded samples and refuses a changed pipeline or dataset
  version. Errors are retried only with `--retry-errors`, since many are deterministic.
- **Attempts.** A retry gets `attempt-2`, so tries never mix. The newest row wins.
- **Failures are results.** `results.json` reports `n_selected`, `n_run`, `n_failed`,
  and each error, so a pipeline can't look better by failing on long consultations.
- **Overall** appears only for metrics in two or more datasets, as an unweighted mean.

## Telemetry

`telemetry/__init__.py` defines what the harness considers when comparing pipelines.
Two kinds of measurement matter, and they answer different questions.

| Measured | By | Answers |
| --- | --- | --- |
| Outside: `wall_seconds` | The harness, around `run` | What a sample costs, the same way for every pipeline |
| Inside: phases, processes, models | The pipeline | Where that cost goes |

Inside data has shared names, so runs compare: `transcription` and `note`, with
`prefill` and `generation` nested under `note`. All of it is optional.

`telemetry/otlp.py` reads OpenTelemetry JSONL, whoever exported it, and `results.json`
adds time per span and machine memory. Each time the pipeline opens it gets its own
folder, so nothing overwrites earlier telemetry.

**Memory figures differ on purpose.** `peak_rss_bytes` is what the pipeline's processes
hold, and it's the number to compare. `machine_memory_bytes` is everything on the
machine, other programs included, and is only context.

`manifest.json` lists the hardware present, including GPUs and NPUs from NVIDIA, AMD,
Apple, and what Linux exposes. Presence isn't use: `Phase.device` records where work
actually ran, and `results.json` summarizes it under `compute`.

## Environments

A pipeline's dependencies are an optional extra. Incompatible stacks, like Scribnic's
`moss` and `moss-audio`, can't share a process, so the environment is chosen at launch.
With one pipeline per run, nothing needs orchestrating. `manifest.json` records versions.

## Language

Metrics use each dataset's language, and pipelines must match it. English results
validate the harness and the pipeline. They say nothing about quality in Spanish.

## Layout

```text
src/scribe_eval/
├── artifacts.py      # audio, transcripts, notes
├── types.py          # Sample, Prediction, Requires, Score, Metric
├── config.py         # TOML → ExperimentConfig
├── selection.py      # split, limit, fraction, ids → sample IDs
├── validation.py     # checks before loading models
├── inference.py      # first phase
├── scoring.py        # second phase
├── aggregation.py    # scores → results
├── reporting.py      # results → report.html
├── storage.py        # run layout, JSONL, predictions
├── hardware.py       # describes the machine
├── application.py    # prepare, run, resume, score
├── cli.py            # Typer
├── terminal.py       # Rich
├── assets/           # report template, styles, charts
├── telemetry/
│   ├── __init__.py   # the contract
│   └── otlp.py       # generic OpenTelemetry reader
├── pipelines/
│   ├── __init__.py   # factory
│   ├── base.py       # the interface
│   └── scribnic/     # the only code that imports Scribnic
├── datasets/
│   ├── prepared.py   # common format and its loader
│   ├── primock57.py  # converter
│   └── teme.py       # converter; builds the smoke dataset
└── metrics/
    ├── transcription.py
    ├── note_format.py
    ├── note_reference.py
    └── resources.py
```

## Planned

- **`compare`,** a table across runs, reading `results.json`.
- **Repeats.** Generating the same input several times shows how much a note changes
  when nothing else does. Without that baseline, a perturbation's effect means little.
  It needs a varying seed and sampling; with a fixed seed and temperature 0, repeats
  add nothing.
- **Perturbations.** An `Artifact -> Artifact` function with a seed, applied before
  the pipeline. Each creates a `variant` compared against `clean`.
- **Judges and entity extraction.** Ordinary metrics. Running at scoring time, they
  don't compete with inference for memory.

`variant` and `repeat` are already in every row, so none of this needs a migration.

## Prior art

- [Inspect](https://inspect.aisi.org.uk/tasks.html): dataset, solver, and scorer kept apart.
- [HELM](https://crfm-helm.readthedocs.io/en/v0.5.3/tutorial/): the run layout.
- [lm-evaluation-harness](https://github.com/arkohut/lm-evaluation-harness/blob/main/docs/new_task_guide.md):
  metric lists per task, and tags.
- [AU-Harness](https://github.com/ServiceNow/AU-Harness): task-metric pairs, and
  sample and global WER. Its engines, batching, and sharding solve a bigger problem.
- [ADS_evaluation](https://github.com/AutoEvaluation/ADS_evaluation): what to measure
  in clinical notes. See [metrics](metrics.md).
