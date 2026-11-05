# Architecture

`scribe_eval` connects data, one Scribnic workflow, and metrics. It depends on
Scribnic and never changes it.

## Principles

1. **One experiment, one workflow.** It can cover several datasets. To compare
   workflows, compare finished runs.
2. **Inference and scoring are separate.** A new metric or judge never regenerates
   notes.
3. **An experiment is a definition; a run is an execution.** `experiments/` holds
   versioned TOML. `runs/` holds what was generated, with sample IDs frozen.
4. **Datasets convert once to a common format.** One loader reads them all.
5. **No pipeline abstraction.** The harness calls `open_workflow` directly. Tests use
   its `backend_factory` with fakes.
6. **Explicit registries.** Datasets and metrics live in plain dictionaries.
7. **The CLI is a frontend.** Everything works from `application.py`.

## Flow

```text
prepare    raw data → data/<dataset>/
validate   experiment + dataset + workflow plan → errors or n/a, no models loaded
infer      sample → session.run → predictions.jsonl
score      prediction + sample → scores.jsonl
aggregate  scores → results.json
```

## Experiments

```toml
name = "primock57-sample"
workflow = "../workflows/whisper_small_bonsai_en.toml"  # preset name or file

[[datasets]]
name = "primock57"
split = "all"
input = "audio"        # what enters the workflow
limit = 3              # or fraction = 0.2, or ids = [...]
seed = 42
metrics = ["transcription", "note_format", "resources"]  # metrics or groups
```

`input` matters: a dataset with audio and transcripts can feed a workflow either
way, and that changes what can be measured. The TOML keeps the intent (`limit`).
`manifest.json` keeps the resolved IDs, so a resume uses the same samples.

## Data structures

These build on Scribnic's artifacts: `AudioFile`, `Transcript`, `DiarizedTranscript`,
and `Note`. See `src/scribe_eval/types.py`.

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
    artifacts: dict[ArtifactKind, Artifact]  # only what the workflow produced
    steps: tuple[dict[str, object], ...]
    resources: dict[str, object]
    elapsed_seconds: float
    run_directory: str    # Scribnic's own output

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

- `Prediction.artifacts` leaves out the input. Scribnic's `RunResult` includes it,
  and a reference transcript fed in must not score a perfect WER.
- A failed run keeps what it produced. If the note is rejected, WER still counts.
- `aggregate` belongs to each metric: global WER is total errors over total words,
  not a mean of per-sample rates. That's why `Score` carries `details`.
- `diarized_transcript` also satisfies `transcript`, through `without_speakers()`.

## Validation

Runs before any model loads. What a workflow produces comes from Scribnic's
`resolve_plan`.

| Case | Result |
| --- | --- |
| Dataset lacks the input or a required reference | Error |
| Workflow doesn't accept the input | Error |
| Workflow doesn't produce the artifact, like WER on `lfm3_audio_note` | `n/a`, with the reason |
| A transcript goes in and WER is requested | `n/a`, with the reason |

## Runs

Laid out like [HELM](https://crfm-helm.readthedocs.io/en/v0.5.3/tutorial/): spec,
per-instance state, per-instance stats, and aggregates in separate files.

```text
runs/20261105-153000-primock57-sample-1a2b3c/
├── experiment.toml      # exact copy
├── manifest.json        # IDs, versions, effective workflow
├── status.json          # running, completed, interrupted, or failed
├── startup.jsonl        # model load times
├── results.json         # dataset × metric, plus overall
└── datasets/primock57/
    ├── predictions.jsonl
    ├── scores.jsonl
    └── samples/<sample_id>/attempt-1/   # Scribnic's output
```

- **JSONL is the source of truth.** One row per sample, flushed as it finishes.
- **Resume** skips recorded samples and refuses a changed workflow or dataset
  version. Errors are retried only with `--retry-errors`, since many are deterministic.
- **Attempts.** Scribnic needs an empty directory, so a retry gets `attempt-2`. The
  newest row for a sample wins.
- **Failures are results.** `results.json` reports `n_selected`, `n_run`, `n_failed`,
  and each error, so a workflow can't look better by failing on long consultations.
- **Overall** appears only for metrics in two or more datasets, as an unweighted mean.

## Environments

Scribnic declares what each workflow needs and marks `moss` and `moss-audio` as
incompatible. Two Transformers versions can't share a process, so the environment is
chosen at launch. With one workflow per run, nothing needs orchestrating here.
`manifest.json` records the versions. Those Scribnic extras aren't exposed by
`scribe_eval` yet.

## Language

Scribnic's presets are Spanish: Whisper with `language = "es"`, notes with `soap_es`.
English datasets use workflows from `workflows/`, which override the language and
point `template` at a local file. English results validate the harness and the
pipeline. They say nothing about quality in Spanish.

## Layout

```text
src/scribe_eval/
├── types.py          # Sample, Prediction, Requires, Score, Metric
├── config.py         # TOML → ExperimentConfig
├── selection.py      # split, limit, fraction, ids → sample IDs
├── validation.py     # checks before loading models
├── inference.py      # first phase
├── scoring.py        # second phase
├── aggregation.py    # scores → results
├── storage.py        # run layout, JSONL, predictions
├── application.py    # prepare, run, resume, score
├── cli.py            # Typer
├── terminal.py       # Rich
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

- **HTML reports and `compare`,** reading `results.json`.
- **Repeats.** Generating the same input several times shows how much a note changes
  when nothing else does. Without that baseline, a perturbation's effect means little.
  It needs a varying seed and sampling; with a fixed seed and temperature 0, repeats
  add nothing. `open_workflow` takes a `WorkflowConfig`, so Scribnic stays untouched.
- **Perturbations.** An `Artifact -> Artifact` function with a seed, applied before
  the workflow. Each creates a `variant` compared against `clean`.
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
