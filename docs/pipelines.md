# Adding a pipeline

A pipeline is anything that turns audio or a transcript into artifacts. To evaluate
yours, implement `Pipeline` from `scribe_eval.pipelines.base` and name it in an
experiment. Datasets, metrics, resume, and reports then work as they are.

```python
class Pipeline(Protocol):
    def describe(self) -> dict: ...                 # effective config, with a "name"
    def produces(self, input_kind) -> set: ...      # what it makes from this input
    def open(self, telemetry_directory): ...        # holds resources; may run per sample
    def prepare(self, input_kind) -> dict: ...      # load models, return load times
    def run(self, source, output_directory) -> PipelineResult: ...
    def timeline(self, telemetry_directories, telemetry) -> Timeline: ...
```

Constructing a pipeline must be cheap: validation needs it before any model loads. It
receives `base_directory` and the settings from the `[pipeline]` table as keyword
arguments.

```toml
[pipeline]
type = "my_scribe.evaluation:MyPipeline"  # or a built-in name, like "scribnic"
model = "large"                           # anything else goes to your constructor
```

## What `run` returns

```python
PipelineResult(
    status="success",
    artifacts={"transcript": transcript, "note": note},
    telemetry=SampleTelemetry(phases=(Phase("note", 12.3),)),
)
```

- Convert to and from `scribe_eval.artifacts`. Keep your own types inside your code.
- Return only what you produced, never the input.
- A failure is a result: return `status="error"` with the artifacts you have and the
  message. Exceptions are for things that should stop the whole run.
- `output_directory` is yours for anything worth keeping.

## Telemetry

Everything is optional. The harness always measures `wall_seconds` itself, so time
comparisons work with no effort. Fill in more to unlock more metrics:

| Field | Unlocks |
| --- | --- |
| `phases` | `phase_seconds` and the time chart |
| `processes`, `peak_rss_bytes` | Memory and CPU metrics |
| `models` | Tokens and tokens per second |
| `Phase.device` | Where each phase actually ran |

Use the shared phase names so runs compare: `transcription`, `note`, and `prefill` and
`generation` with `parent="note"`.

`timeline` feeds the report's memory chart. Return an empty `Timeline()` if you have
nothing. Traces and metrics written as OTLP JSONL to `telemetry_directory` also get
summarized in `results.json`, whatever exported them.

## Rules

- `open` can run once per sample, so closing must release everything. Anything left
  running would break isolation.
- Your package is imported only by your implementation. Built-ins live in
  `src/scribe_eval/pipelines/` and are optional extras.
- A resumed run must `describe` itself identically, or resuming is refused.

`pipelines/scribnic/` is a complete example, and `FakePipeline` in `tests/helpers.py`
is the minimal one.
