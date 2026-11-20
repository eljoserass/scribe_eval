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

## Berta AI Scribe

`type = "berta"` evaluates [Berta AI Scribe](https://github.com/phairlab/berta-ai-scribe)
(Apache-2.0) through its own tasks: `transcribe_audio` with its WhisperX service, and
`generate_note` with its plain-text note prompt, its default `Full Visit` note type,
and its llama.cpp adapter. Berta has no diarization, so it makes a transcript and a
note.

### Setup

Berta runs from a checkout of the [fork](https://github.com/eljoserass/berta-ai-scribe)
in its own Python 3.11 environment. The note model is the one Scribnic's English
workflow uses, so a Scribnic checkout must sit next to this repository.

```bash
scripts/setup_berta.sh               # BERTA_DIR, SCRIBNIC_DIR, BERTA_TORCH_BACKEND
uv run scribe-eval run experiments/primock57/berta_smoke.toml
```

The script needs `uv`, `ffmpeg`, and `git`, on Linux or macOS on Apple Silicon. It:

1. Clones the fork into `../berta-ai-scribe` unless `BERTA_DIR` points to one.
2. Creates `web-api/.venv` there with Python 3.11 and installs Berta's
   `requirements.txt`, with CPU PyTorch by default.
3. Downloads faster-whisper `small`.
4. Finds the Prism llama-server and `Ternary-Bonsai-2-2B-PQ2_0.gguf` in Scribnic's
   cache, running `scribnic setup --workflow bonsai_note` if they are missing.
5. Links Berta's `web-api`, llama-server, and the model in `vendor/berta/`, which
   the experiments use, and converts PriMock57 unless it already is.

### Settings

| Setting | Default | Meaning |
| --- | --- | --- |
| `berta_directory` | required | Berta's `web-api` directory |
| `llama_server` | required | llama-server binary for the note model |
| `note_model` | required | GGUF file; served under its file name |
| `python` | `.venv/bin/python` in `berta_directory` | Berta's interpreter |
| `whisper_model` | `small` | faster-whisper model for Berta's WhisperX service |
| `whisper_compute_type` | `float32` | CTranslate2 compute type; Berta's own is `float32` |
| `whisper_device` | `cpu` | Berta's `WHISPERX_DEVICE`: `cpu`, `cuda`, or `cuda:N` |
| `note_type` | `Full Visit` | One of Berta's built-in note types |

Relative paths start at the experiment's directory. `describe()` records them
resolved, with Berta's commit, both models, and llama-server's arguments.

### How it runs

Berta can't share scribe_eval's process, so it has no extra here. It configures its
services when `app.config.ai` is imported, as module globals, so a model loaded in
the harness would stay loaded after `close`. It reads prompts through paths that must
be relative to `web-api`. Its pins (`pydantic==2.9.2`, `whisperx`, `torch~=2.8`)
resolve on Python 3.12 but not on 3.15, which scribe_eval supports and CI tests.

Instead, `prepare` starts two processes, and the end of `open` stops both:

- **llama-server** on a free loopback port, with its log in the telemetry directory.
- **Berta's worker**, `pipelines/berta/worker.py`, run by Berta's Python in
  `web-api`. It sets Berta's settings through the environment, imports Berta once,
  and answers one JSON line per sample over stdin and stdout. Its stderr goes to
  `berta-worker.log` in the telemetry directory.

The worker changes one thing in Berta: its WhisperX service hardcodes `large-v3` and
`float32`, and the worker replaces those two values before the service is built. It
transcribes as Berta's web app does: the recording is converted to MP3 at Berta's
default 96 kbit/s, as on upload, then transcribed. That conversion counts in the
`transcription` phase. Berta's recordings folder and database file point to a
temporary directory, so its checkout isn't written to. Model downloads are off
(`HF_HUB_OFFLINE=1`), and no cloud service is configured; Berta still reads
`web-api/.env`, but the settings the pipeline sets take precedence.

A failed task is a result: the worker reports the error with the transcript it
already has. If a process dies, that sample fails and the next one stops the run;
`resume --retry-errors` starts them again.

Telemetry has `transcription` and `note` from the worker, and `prefill` and
`generation` from llama-server's counters before and after each sample. Tokens come
from Berta's reply. Memory and CPU of both processes are sampled every 0.25 s and
saved for the report's chart in `berta-timeline.jsonl`. Each sample's
`pipeline/berta.json` keeps the worker's reply.

### Compared with Scribnic

`experiments/primock57/berta_*.toml` borrow Scribnic's models: Whisper small for transcription,
and the same note model on the same llama-server build and options. That makes them a
hybrid. They are not Berta out of the box, whose defaults are Parakeet MLX for
transcription and `llama3.1:8b` through Ollama for notes. Nor are they a controlled
comparison: the speech runtime, sampling, and limits differ, as below.

| | Scribnic `whisper_small_bonsai_en` | Berta |
| --- | --- | --- |
| Speech runtime | whisper.cpp, `small` q8_0 | WhisperX 3.8.6, faster-whisper 1.2.1, `small` float32, Pyannote VAD, batch size 1 on CPU, 4 threads |
| Language | `en` | Detected from the first 30 s |
| Audio | WAV as prepared | MP3 at 96 kbit/s first |
| Note prompt | `soap_en.md`: four SOAP sections, at most 250 words | Plain-text system prompt and `Full Visit`: an emergency department note with chief complaint, history, exam, and impression and plan |
| Note model | Ternary-Bonsai-2-2B-PQ2_0, Prism llama-server b10787 | Same |
| Server | 8192 context, 1 slot, 6 threads, offload on macOS only, no prompt cache, no thinking | Same |
| Sampling | Temperature 0, Prism's profile (top-p 0.8, top-k 20, presence penalty 1.5, seed 42), 768 tokens | Temperature 0 and 4096 tokens, set in Berta's adapter; server defaults otherwise |
| Long input | Refused above 12,000 characters | Sent whole |
| Output checks | Truncated or thinking output fails | None |

Notes from the two pipelines don't share a template, so reference-note metrics
compare the scribes as a whole, prompt included.

### Limits

- Berta's plain-text prompt appends the context field after the transcript. Its web
  app sends none, so the prompt ends with the word `None`, as in Berta.
- Berta's built-in note types are stored as Markdown in its database. Plain text is
  a per-note-type option in Berta, and the pipeline always uses it, since Markdown
  symbols would count in overlap metrics. Nothing is stripped from the output.
- Transcripts are one string, with no segments or times. A diarized input loses its
  speakers, as Berta's own transcripts have none.
- WhisperX runs on CPU on macOS: CTranslate2 has no Metal backend.
- A worker hang has no timeout. Berta's llama.cpp adapter gives up after 300 s.
- The setup leans on Scribnic: `scripts/setup_berta.sh` takes llama-server and the
  Bonsai GGUF from Scribnic's cache, and runs `scribnic setup` when they are missing.
- The pipeline always starts its own worker and llama-server. An already running Berta
  or LLM server can't be used; it would also leave its memory out of the telemetry.

### Verified

On Linux (WSL2, x86_64, CPU), `experiments/primock57/berta_smoke.toml` ran end to end on
`day1_consultation01` and `day1_consultation03`, with Berta 0.6.0 at commit
`87fe138`, WhisperX 3.8.6, and Prism llama-server b10787. Both succeeded, and a second
run gave the same transcripts and notes. Without other load on the machine:

| Consultation | Time | Transcription | Note | WER | Note words |
| --- | --- | --- | --- | --- | --- |
| `day1_consultation01` | 138.6 s | 115.9 s | 22.6 s | 0.176 | 106 |
| `day1_consultation03` | 151.1 s | 126.6 s | 24.5 s | 0.175 | 111 |

Peak memory of both processes together was 3.0 GB.

On macOS 15.6 (Mac Studio, M3 Ultra), `experiments/primock57/berta_full.toml` ran all 57
consultations with no failed rows: 179 s per consultation, 176 s of it transcription on
CPU; global WER 0.172; peak memory 6.7 GB. Four notes ran past 1,000 words (at most
2,674), repeating plan lines until the 4,096-token limit, and one listed dialysis and
an orthopedics follow-up that the consultation doesn't mention. Berta accepts these
notes, so no failure rows flag them.
