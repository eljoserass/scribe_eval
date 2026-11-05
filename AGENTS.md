# Working on scribe_eval

`scribe_eval` is the glue between datasets, a scribe pipeline, and metrics. Keep the
core easy to read, navigate, test, and reuse from Python and the CLI. The style rules
follow Scribnic's `AGENTS.md`. Prefer an easy-to-maintain design over an optimal one.
See `docs/architecture.md`.

## Writing

- Write code, comments, and documentation in English.
- Write docs for someone reading the project, not for whoever asked for them. Say
  what the system does and why. Don't address the maintainer or describe the
  conversation that led to a change.
- Keep it terse and human. Prefer short sentences and concrete facts. Cut filler,
  hedging, and sections that repeat each other. If a table or list says it, don't
  say it again in prose.
- State what was measured and how. Mark what is unverified.

## Reference guides and engineering review

- Use the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  for Python language and style decisions, and Google's
  [code review guidance](https://google.github.io/eng-practices/review/reviewer/looking-for.html)
  for design, complexity, concurrency, tests, and documentation. The explicit
  scribe_eval rules below take precedence, including Ruff, 88 columns, and documenting
  private functions.
- Review responsibilities and resource ownership as well as formatting. Keep
  abstractions tied to a concrete current need. A shorter file count or a passing
  linter does not establish a simpler or correct design.
- For concurrent or streaming code, review buffer growth, backpressure, normal
  completion, cancellation, worker shutdown, and failure propagation explicitly.
  Silence, end of an input source, and cancellation are separate events.

## Readable Python

- Follow PEP 8 with four-space indentation and an 88-character formatting target.
  Use Ruff for formatting, import sorting, and linting when adding development
  tooling. Development tools do not belong in client runtime dependencies.
- Give each module one clear responsibility and each function one coherent task.
  Prefer concrete names that explain the domain and include units where relevant.
- Aim for functions below 50 body lines, excluding blank lines, comments, and
  docstrings. Crossing that threshold is a reason to review the responsibility,
  not to compress statements or extract meaningless helpers. Keep a longer
  function when its cohesive sequence is easier to follow, and explain why.
- Put executable statements on separate lines. Avoid semicolon-separated
  statements and inline bodies for `if`, `for`, `with`, and function definitions.
  Protocol declarations and typing stubs may use conventional ellipsis bodies.
- Use comprehensions for simple mappings or filters. Use ordinary loops when
  there are nested iterations, several conditions, side effects, or error handling.
- Avoid nested conditional expressions and long expressions with several jobs.
  Introduce named intermediate values when they clarify the calculation.
- Prefer guard clauses and early returns to deep nesting. Ordinary `else` and
  `elif` branches are fine when they make the alternatives clearer.
- Keep whitespace between logical blocks. Let the formatter handle wrapping;
  never squeeze code to satisfy a line-count guideline.
- Add type annotations to function parameters, return values, and shared data.
  Avoid `Any` when a concrete type or a small Protocol describes the contract.
- Document behavior, assumptions, units, and non-obvious constraints using the
  function documentation rules below. Comments should explain decisions rather
  than repeat the adjacent statement.
- Catch specific exceptions at the boundary that can handle them. Preserve the
  original cause and provide actionable context. Avoid blanket exception handling
  and silent fallbacks.
- Use context managers or `try`/`finally` for resource ownership and cleanup.

## Function signatures and documentation

- Give every named function and method a docstring, including private helpers
  and constructors. A short, obvious function needs only a one-line summary.
  Apply this when adding or modifying a function; do not launch an unrelated
  documentation sweep.
- Use Google-style docstrings: start with what the function does, then add
  `Args`, `Returns`, `Yields`, and `Raises` sections where they clarify its
  contract. Explain parameter meaning, units, accepted values, and important
  defaults; keep types in annotations rather than repeating them in prose.
- Document observable side effects and ownership: whether a call downloads,
  loads or retains a model, starts a process, writes files, or closes a resource.
  Describe errors callers need to handle, not every incidental library exception.
- Put the main input first. Make optional behavior switches and tuning settings
  keyword-only with `*`, especially booleans and numeric values with similar types.
  Use named arguments at call sites when their meaning would otherwise be unclear.
- Keep signatures focused on the operation. Group related settings in an existing
  typed configuration object when that improves clarity. Avoid both long lists of
  unrelated options and catch-all configuration objects passed through every layer.
- Use explicit defaults and avoid mutable default arguments. Distinguish `None`
  from valid false or zero values. Prefer names with units such as `timeout_seconds`.
- Let Ruff wrap long signatures and calls with one argument per line and a trailing
  comma. Use modern annotations such as `list[str]` and `str | None`.

For example:

```python
def real_time_factor(
    processing_seconds: float,
    *,
    audio_seconds: float,
) -> float:
    """Calculate processing time per second of source audio.

    Args:
        processing_seconds: Elapsed processing time, excluding model preparation.
        audio_seconds: Positive duration of the original recording.

    Returns:
        The time ratio; values below one mean processing faster than real time.

    Raises:
        ValueError: If the recording duration is not positive.
    """
    if audio_seconds <= 0:
        raise ValueError("Audio duration must be positive.")
    return processing_seconds / audio_seconds
```

## Logical organization

- Give modules a short docstring explaining their responsibility. Group related
  constants, types, and operations so readers can find the public entry points and
  follow their helpers without jumping between unrelated concerns.
- Separate cohesive phases with blank lines. A short comment such as
  `# Prepare the input` can name a phase when it helps readers follow a longer
  sequence. Avoid decorative banners and comments that narrate every statement.
- Group configuration fields by purpose and document units or constraints near
  the fields. Keep CLI parsing, model preparation, inference, and persistence in
  their respective layers even when a script presents them as consecutive phases.
- A phase that owns separate state, error handling, or reusable behavior is a
  candidate for a helper. A section comment does not make a large function cohesive.
- Pass dependencies explicitly or keep them on the owning session. Avoid mutable
  module globals for models, clients, inference contexts, and per-run state.

## Functions, classes, and module boundaries

- Give every new or modified class a docstring stating its responsibility,
  owned resources, and lifetime. Document behavior rather than repeating names.
- The harness knows nothing about any particular scribe. A scribe implements
  `Pipeline` in `pipelines/`, and only that implementation may import its packages.
  Dataset loading, perturbations, judges, and metrics live in the harness.
- Metrics read the telemetry contract, never a pipeline's own fields. Add to the
  contract when something is worth comparing across pipelines.
- A converter reads one raw dataset format and writes the common prepared format.
  The prepared loader knows nothing about individual datasets.
- A metric declares what it requires and how it aggregates. Inference code must not
  know about metrics, and metrics must not run workflows.
- Keep CLI responsibilities to argument parsing, invoking `application.py`, and
  presenting results or errors. Everything must work without the CLI.
- Keep persistence, report rendering, and terminal presentation separate from
  inference and scoring.
- Use dataclasses for configuration and records. Prefer immutable records; create
  explicit copies for transformations. Use classes for stateful resources and
  functions for transformations and calculations.
- Use small Protocols where implementations need a shared interface. Register
  datasets and metrics explicitly in dictionaries; avoid decorators, autodiscovery,
  factories, and inheritance unless they simplify a concrete use.

## Experiment and result contracts

- One experiment evaluates one pipeline, on one or more datasets. Each dataset entry
  selects its own input and metrics. Compare pipelines by comparing finished runs.
- Validate datasets, requested inputs, and metric requirements before loading models.
  A missing dataset artifact is an error. A metric the pipeline cannot satisfy is
  `n/a` with a visible reason.
- Freeze the resolved sample IDs in the run manifest. Resuming must use the same
  samples and configuration, or fail clearly.
- Treat JSONL rows as the source of truth, appended once per completed sample. Keep
  `status.json` as a summary. A row key is `dataset, variant, sample_id, repeat`.
- Failures are results. Record them as rows, keep partial artifacts, and report
  failure counts beside every aggregate. Never drop them silently.
- `Prediction.artifacts` contains only what the pipeline produced, never its input.
- Metric normalization is part of the metric. Version it, and apply it equally to
  prediction and reference.
- Record the effective pipeline configuration, versions, and hardware so a run can
  be interpreted later. Keep model loading separate from inference timing.
- Do not present a result as clinical validation. Record dataset language and
  provenance, and keep synthetic or fixture references labeled as such.

## Data handling

- Datasets, runs, and recordings are not source code; keep them out of git.
- Keep the licence and source of each dataset in `dataset.json` and `docs/datasets.md`.
- Do not send dataset content to external services without an explicit decision.
  Clinical data must stay local; a remote judge is only for public or synthetic data.

## Dependencies and environments

- Keep the base installation small. Import heavy metric libraries inside the metric
  that uses them, behind optional extras. Importing the package must not need Torch.
- Pipeline dependencies are optional extras. Importing `scribe_eval` must not import
  any pipeline package. Choose environments when launching a run; do not orchestrate
  them here.

## Verification and repository hygiene

- Run `uv run ruff check src tests` and `uv run ruff format --check src tests`.
  Ruff handles layout; do not add competing formatters.
- Run `uv run python -m unittest discover -t . -s tests`.
- Test meaningful behavior and public boundaries with small fakes. Test the core with
  a fake `Pipeline`, and each implementation separately. Cover validation, sample
  selection, resume, failure rows, and aggregation.
- Keep live model checks and dataset downloads opt-in. Ordinary unit tests must not
  download weights or data, require a GPU, or contact inference services.
- Run checks appropriate to the change. Documentation-only edits need no model runs.
