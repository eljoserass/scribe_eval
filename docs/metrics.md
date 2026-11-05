# Metrics

Each metric declares what it needs and how it aggregates. Experiments can name a
metric or a whole group. Nothing here measures clinical safety yet.

## Available

| Group | Metric | Predicted | Reference | Aggregates |
| --- | --- | --- | --- | --- |
| `transcription` | `wer`, `cer` | `transcript` | `transcript` | Global, sample mean |
| `note_format` | `soap_sections` | `note` | — | Rate |
| `note_format` | `note_words` | `note` | — | Mean, max |
| `note_reference` | `rouge_l` | `note` | `note` | Mean |
| `resources` | `elapsed_seconds` | — | — | Mean, p95 |
| `resources` | `real_time_factor` | — | `audio` | Global |
| `resources` | `peak_rss_bytes` | — | — | Max |
| `resources` | `phase_seconds` | — | — | Mean per phase |
| `resources` | `phase_real_time_factor` | — | `audio` | Global per phase |
| `resources` | `process_peak_rss_bytes` | — | — | Max per process |
| `resources` | `process_cpu_cores` | — | — | Mean per process |
| `resources` | `prefill_tokens_per_second`, `generation_tokens_per_second` | — | — | Mean |
| `resources` | `prompt_tokens`, `completion_tokens` | — | — | Total, mean |

## Rules

- **Global and sample mean.** Global WER is total errors over total words. The sample
  mean gives a short consultation the same weight as a long one.
- **Normalization is part of the metric.** Lowercase, spelled-out numbers, no
  punctuation, no hesitations, in the sample's language. Human transcripts keep every
  "um" and ASR drops them, which isn't an error: on two PriMock57 consultations they
  added about 3.6 points of WER. Changing normalization means bumping the metric's
  `version`, which `manifest.json` records.
- **Failed runs.** Artifact metrics still score what exists. Metrics that need no
  artifact only count successful runs. `n_failed` always sits next to the numbers.
- **Same language.** A reference metric means nothing if the note and the reference
  are in different languages.
- **`soap_sections`** checks that S, O, A, and P appear in order with nothing before
  them. It can't see junk after the plan, like the `No consta | ...` line Bonsai 2B
  sometimes adds.
- **`resources`** read the [telemetry contract](architecture.md#telemetry), so they
  work for any pipeline that fills it in. Only `elapsed_seconds` and `real_time_factor`
  work for every pipeline, because the harness measures them. The rest are `n/a` when
  a pipeline doesn't report them. Memory is sampled RSS of the pipeline's processes,
  not the whole machine, and may double count shared pages. For Scribnic that includes
  the client, about 0.12 GB. Model loading is in `startup.jsonl`, not here.
- **Peak memory and order.** Unless `isolate_samples = true`, a sample inherits what
  earlier ones left, so peaks mix consultation length with order. The report's RAM
  start and end columns show what stayed.

## ROUGE is a diagnostic

On a real Bonsai 4B note, with hand-made mutations, a correct paraphrase scored 0.887
ROUGE-L. Wrong doses, flipped negations, and wrong severity scored 0.984–0.997. One
case, but enough to never read it as a safety signal.

## What ADS_evaluation does

[ADS_evaluation](https://github.com/AutoEvaluation/ADS_evaluation) is the code behind
Wang et al. 2025, and the closest reference for what to measure. It's scripts and
notebooks, not a harness.

**Automatic:** ROUGE-1/2/L/Lsum, BLEU, METEOR, BERTScore, GPT-2 perplexity, and a
toxicity classifier.

**Facts** (the most useful part):

1. An LLM extracts atomic facts from the generated and reference notes: one piece of
   information each, self-contained, figures and units intact.
2. Each predicted fact is checked against the reference facts, by an LLM or by NLI
   (`roberta-large`, entailment above 0.95).
3. `correct_score` = supported / predicted facts. Roughly precision.
4. `link_score` = min(1, supported / reference facts). A loose stand-in for recall.

`link_score` isn't real recall: it counts predicted facts, not covered reference
facts, and several can lean on the same one. Coverage should be computed the other way.

**LLM judge.** Two modes, against the transcript or a reference note. Each dimension
gets Yes/No, 1–5, and an explanation.

| Block | Dimensions |
| --- | --- |
| Linguistic quality | Fluency, coherence, clarity, brevity, structuring |
| Content integrity | Relevance, completeness, factuality, comprehension |
| Trustworthiness | Prudence, toxicity, bias, fairness |

**Analysis.** Human-judge agreement, including Cohen's kappa, and 5 generations per
transcript to see variation.

To adopt any of this:

- It calls GPT-4 and GPT-4o at temperature 0.7. Clinical data needs a local judge, and
  a small model has to prove it can do the job.
- Rubrics are in English and need translating and validating for Spanish.
- A parsing failure counts as `NO`. Here it should be a visible error.
- Validate judges before trusting them, for instance against PriMock57's human data.

[EvalLLM4Health](https://github.com/AutoEvaluation/EvalLLM4Health) is an earlier repo
from the same group. It adds a human scorecard, with the same Yes/No and 1–5 rubric,
and a table of adversarial tests.

## Planned

| Kind | Idea | Needs | From |
| --- | --- | --- | --- |
| Facts | Precision and coverage of atomic facts | LLM or NLI | ADS_evaluation |
| Negation | Entities as present, absent, or unknown | Entity extractor | CliniKnote |
| Judge | The 13 dimensions, in both modes | Validated local LLM | ADS_evaluation |
| Critical errors | Critical omissions and false statements | Judge validated on human data | PriMock57 |
| Medication | Drugs, doses, and coherence, with severity | Rules and a judge | TEME, AAMAS 2026 |
| Semantic | BERTScore, BLEURT, QuestEval | Extra models, Torch | ADS_evaluation, CliniKnote |
| Diarization | DER, WDER, cpWER, speaker count error | Design pending | [AU-Harness](https://github.com/ServiceNow/AU-Harness) |

The judge's two modes fit `Requires`: one needs a `note` reference, the other a
`transcript`.

## Planned perturbations

From EvalLLM4Health and ADS_evaluation:

- Mask 10% of the transcript, or specific data like lab dates.
- Swap speakers.
- Insert irrelevant conversation or inappropriate language.
- Insert impossible lab values.
- Use rare diseases or new drugs.
- Add noise or cuts to the audio.
