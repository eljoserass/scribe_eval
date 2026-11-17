# Metrics

Each metric declares what it needs and how it aggregates. Experiments can name a
metric or a whole group. Nothing here measures clinical safety yet.

## Available

| Group | Metric | Predicted | Reference | Aggregates |
| --- | --- | --- | --- | --- |
| `transcription` | `wer`, `cer` | `transcript` | `transcript` | Global, sample mean |
| `transcription` | `medical_term_wer` | `transcript` | `transcript` | Global |
| `transcription` | `transcript_rouge_1`, `transcript_rouge_l` | `transcript` | `transcript` | Mean |
| `transcription` | `transcript_bleu` | `transcript` | `transcript` | Corpus, mean |
| `transcription_semantic` | `clinical_bertscore` | `transcript` | `transcript` | Global, sample mean |
| `diarization` | `der` | `diarized_transcript` | `diarized_transcript` | Global, by cause |
| `diarization` | `jer` | `diarized_transcript` | `diarized_transcript` | Mean |
| `note_format` | `note_sections` | `note` | — | Rate, coverage |
| `note_format` | `note_words` | `note` | — | Mean, max |
| `note_format` | `flesch_reading_ease` | `note` | — | Mean |
| `note_format` | `compression_ratio` | `note` | `transcript` | Global |
| `note_reference` | `rouge_1`, `rouge_2`, `rouge_l` | `note` | `note` | Mean |
| `note_reference` | `bleu` | `note` | `note` | Corpus, mean |
| `note_reference` | `meteor` | `note` | `note` | Mean |
| `note_reference` | `levenshtein` | `note` | `note` | Mean, per character |
| `note_semantic` | `bertscore` | `note` | `note` | Mean, with precision and recall |
| `note_semantic` | `bleurt` | `note` | `note` | Mean |
| `link_correct` | `link`, `correct` | `note` | `note` | Mean |
| `note_grounding` | `grounding` | `note` | `transcript` | Mean, pooled |
| `resources` | `elapsed_seconds` | — | — | Mean, p95 |
| `resources` | `real_time_factor` | — | `audio` | Global |
| `resources` | `peak_rss_bytes` | — | — | Max |
| `resources` | `phase_seconds` | — | — | Mean per phase |
| `resources` | `phase_real_time_factor` | — | `audio` | Global per phase |
| `resources` | `process_peak_rss_bytes` | — | — | Max per process |
| `resources` | `process_cpu_cores` | — | — | Mean per process |
| `resources` | `prefill_tokens_per_second`, `generation_tokens_per_second` | — | — | Mean |
| `resources` | `prompt_tokens`, `completion_tokens` | — | — | Total, mean |

Reproducibility comes first: metrics use the standard library for each measure.
Every metric fails somewhere, so read several together.

## Coverage of the ADS framework

Wang et al.'s Table 2, the reference for what to measure.

| Aspect | Their metrics | Status |
| --- | --- | --- |
| Transcription | WER, ROUGE, BLEU | Done |
| Diarization | DER, JER | Done, with the `diarization` extra. Not tried on a real run yet |
| Diarization | Confusion matrix | Planned |
| Fluency | Perplexity (GPT-2) | Skipped for now: fluency isn't the risk in a clinical note |
| Coherence | BERTScore | Done, with the `semantic` extra |
| Clarity | Flesch reading ease | Done |
| Brevity | Compression ratio | Done. They don't define it; this is note words over conversation words |
| Structuring | Structural precision/recall | Skipped: neither the paper nor its code defines it. `note_sections` covers part of it |
| Relevance, completeness | ROUGE, BLEU, METEOR, BERTScore | Done |
| Factuality | LINK, CORRECT | Done, replicating their code, with a swappable judge |
| Factuality without a reference note | — | `grounding`, this project's prompts. Not validated |
| Toxicity | toxic-bert | Skipped for now: near zero on clinical notes |
| Human, LLM judges, simulations | Scorecards, rubrics, perturbations | Later |

## Reproducing published numbers

Checked against the 285 notes scored in PriMock57's `metric-scores.json`.

| Metric | Identical values | Rank correlation |
| --- | --- | --- |
| `levenshtein` | 285 | 1.00 |
| `bleu` | 285 | 1.00 |
| `rouge_l` | 128 | 0.99 |
| `rouge_2` | 167 | 0.99 |
| `rouge_1` | 81 | 0.99 |
| `meteor` | 38 | 0.82 |

ROUGE differs slightly in tokenization. METEOR differs more, and its mean is 0.44 against
their 0.33, so it isn't comparable across implementations.

## Rules

- **Global and sample mean.** Global WER is total errors over total words. The sample
  mean gives a short consultation the same weight as a long one. Global WER also
  splits into substitution, deletion, and insertion rates.
- **Normalization is part of the metric,** applied equally to prediction and
  reference. It follows the Open ASR Leaderboard, through Whisper's normalizers,
  copied verbatim from a pinned commit:
  - English: `EnglishTextNormalizer`. Lowercase, no punctuation, no hesitations,
    numbers as digits, American spelling.
  - Other languages: `BasicTextNormalizer`, keeping accents, after `num2words` spells
    digits out. No standard list of hesitations exists outside English, so they stay.

  Human transcripts keep every "um" and ASR drops them, which isn't an error: on two
  PriMock57 consultations they added about 3.6 points of WER. Changing normalization
  means bumping the metric's `version`, which `manifest.json` records.
- **Failed runs.** Artifact metrics still score what exists. Metrics that need no
  artifact only count successful runs. `n_failed` always sits next to the numbers.
- **Same language.** A reference metric means nothing if the note and the reference
  are in different languages.
- **`note_sections`** checks that the expected sections appear in order with nothing
  before them, and `coverage` reports the share of sections found. A section counts
  when one of its headings starts a line, with or without Markdown. The default is
  SOAP in English, with `S`, `O`, `A`, and `P` also accepted; any other language or
  template is a setting. It can't see junk
  after the last section, like the `No consta | ...` line Bonsai 2B sometimes adds.

  ```toml
  [metric_settings.note_sections]
  sections = [["Motivo de consulta"], ["Exploración"], ["Juicio clínico"], ["Plan"]]
  ```
- **`resources`** read the [telemetry contract](architecture.md#telemetry), so they
  work for any pipeline that fills it in. Only `elapsed_seconds` and `real_time_factor`
  work for every pipeline, because the harness measures them. The rest are `n/a` when
  a pipeline doesn't report them. Memory is sampled RSS of the pipeline's processes,
  not the whole machine, and may double count shared pages. For Scribnic that includes
  the client, about 0.12 GB. Model loading is in `startup.jsonl`, not here.
- **Peak memory and order.** Unless `isolate_samples = true`, a sample inherits what
  earlier ones left, so peaks mix consultation length with order. The report's RAM
  start and end columns show what stayed.

## Medical terms

`medical_term_wer` follows [MedWER](https://github.com/Nordis-Tech/medwer): keep only
terms from a fixed list, then align. A missed phrase is one error. It uses MedWER's list
with Whisper's normalization, not theirs, so numbers aren't comparable. English
only; other languages are skipped. It's precise but misses most serious errors. See
[evaluation notes](evaluation_notes.md).

## LINK and CORRECT

`link` and `correct` replicate Wang et al.'s fact metrics, as in
[ADS_evaluation](https://github.com/AutoEvaluation/ADS_evaluation). A judge extracts facts
from the generated and reference notes, then checks each generated fact, one call each.

```text
CORRECT = supported / generated facts
LINK = min(1, supported / reference facts)
```

Their repository has no license, so its prompts aren't copied here. They're fetched
verbatim from a pinned commit on first use, into `~/.cache/scribe_eval`.

Differences from their setup: the judge model is ours to choose, at temperature 0 instead
of 0.7. The server enforces the JSON shape, and a bad answer fails scoring instead of
counting as `NO`. Compare scores only between runs with the same judge.

The judge is any OpenAI-compatible server, set in `[judge]`. It must be local unless
`allow_remote = true`, since prompts contain clinical text. Answers are cached in
`judge_cache.jsonl`, so `scribe-eval score` costs nothing the second time.

```bash
llama-server -m Qwen3.5-4B-Q8_0.gguf --port 8090 --alias judge -c 16384 --cache-ram 0
```

## Grounding

`grounding` asks how much of a note the reference transcript supports, in the style of
[FActScore](https://arxiv.org/abs/2305.14251) and DocLens attribution. It needs no
reference note, so it applies to TEME-v1. A judge splits the note into atomic facts,
then checks each against the whole transcript, one call each.

```text
grounding = supported facts / extracted facts
```

- **Prompts** are this project's, in English and Spanish, chosen by the dataset's
  language. Another language fails validation. Changing a prompt means bumping
  `version`.
- **Speakers** are shown when the reference has them, so a symptom the doctor asks
  about isn't read as one the patient has.
- **The transcript leads each verdict prompt,** so a server's prompt cache reuses it
  across one note's facts. Keep the cache on for the judge, unlike for the pipeline.
- **Aggregates:** `mean` weighs each note equally; `pooled` weighs each fact.
- **Limits.** It measures precision, not completeness: a short, vague note scores
  well. `note_words` should sit beside it. Against the reference transcript, an
  unsupported fact may come from the note step or from transcription errors; the
  attribution metric planned below separates them. A note with no facts gets no score.
- **Not validated.** Neither the prompts nor any judge model have been checked against
  human judgments. PriMock57's incorrect-statement annotations are the planned check.


## Model metrics

`bertscore`, `bleurt`, and `clinical_bertscore` run neural models, so they need the
`semantic` extra: `uv sync --extra semantic`. Weights download from Hugging Face the
first time a run scores with them, into its cache. Set `HF_HOME` to move it, for
instance to shared storage on a cluster, and `HF_HUB_OFFLINE=1` on nodes without
internet once it's filled. `device = "auto"` picks CUDA, then Apple MPS, then CPU.

Defaults reproduce each paper; `[metric_settings]` changes them per experiment, and
`manifest.json` records the effective model, layer, and package version.

```toml
[metric_settings.bertscore]
model = "microsoft/deberta-xlarge-mnli"  # default: roberta-large in English
device = "cuda"

[metric_settings.bleurt]
checkpoint = "lucadiliello/BLEURT-20-D12"  # default: BLEURT-20, about 2 GB

[metric_settings.clinical_bertscore]
medical_weight = 0.4                       # the paper's tuned k
terms = "~/terms/medical_terms_es.txt"     # one term per line
```

| Metric | Default | Notes |
| --- | --- | --- |
| `bertscore` | `roberta-large`, layer 17, in English; multilingual BERT otherwise | [Zhang et al. 2020](https://arxiv.org/abs/1904.09675), through their `bert-score` package. F1, no IDF, no rescaling: what ADS_evaluation gets from Hugging Face `evaluate` with `lang="en"` |
| `bleurt` | BLEURT-20 | [Sellam et al. 2020](https://arxiv.org/abs/2004.04696). Google's code needs TensorFlow, so this uses the `bleurt-pytorch` port and its conversions of the official checkpoints. Not comparable across checkpoints |
| `clinical_bertscore` | BERTScore's model, k = 0.4, MedWER's term list | [Shor et al. 2023](https://arxiv.org/abs/2303.05737), adapted below |

Inputs past 512 tokens are truncated, and `truncated` counts them. Generated notes
can reach that; transcripts always would, hence the split below.

**Clinical BERTScore** is `k × BERTScore_medical + (1 − k) × BERTScore_all`, where the
medical part weighs only tokens of medical terms and numbers. The paper released no
code, so three things differ:

- Its 20K medical words come from Google's Knowledge Graph and aren't public. MedWER's
  19K-term list stands in, so English only unless `terms` points to another list.
- It scored single sentences. Here each reference utterance is paired with the
  predicted words aligned to it, as for WER, and the sample weighs utterances by
  reference words.
- Text is normalized as for WER. Numbers count as medical, as digits or spelled out.

For Spanish, TEME's authors used `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
with k = 0.4; `model` and `terms` reproduce that setup given a term list.

The paper's clinician preferences (CTP) are public, so the adaptation can be checked
against its reported agreement. Not done yet.

The `bertscore_global` detail is plain BERTScore over the same utterance pairs.

## ROUGE is a diagnostic

On a real Bonsai 4B note, with hand-made mutations, a correct paraphrase scored 0.887
ROUGE-L, with an earlier implementation. Wrong doses, flipped negations, and wrong severity scored 0.984–0.997. One
case, but enough to never read it as a safety signal.

## What ADS_evaluation does

[ADS_evaluation](https://github.com/AutoEvaluation/ADS_evaluation) is the code behind
Wang et al. 2025, and the closest reference for what to measure. It's scripts and
notebooks, not a harness.

**Automatic:** ROUGE-1/2/L/Lsum, BLEU, METEOR, BERTScore, GPT-2 perplexity, and a
toxicity classifier.

**Facts.** LINK and CORRECT, replicated here. `link_score` isn't real recall: it counts
generated facts, not covered reference facts, and several can lean on the same one.

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
- Validate judges before trusting them, for instance against PriMock57's human data.

[EvalLLM4Health](https://github.com/AutoEvaluation/EvalLLM4Health) is an earlier repo
from the same group. It adds a human scorecard, with the same Yes/No and 1–5 rubric,
and a table of adversarial tests.

## Planned

Published metrics first; anything adapted says so.

| Kind | Metric | Needs | From |
| --- | --- | --- | --- |
| Medical concepts | MEDCON: F1 over UMLS concepts | QuickUMLS and a UMLS licence | [ACI-BENCH](https://arxiv.org/abs/2306.02022) |
| Transcription | Check `clinical_bertscore` against the paper's clinician benchmark, CTP | Its public data | [Shor et al. 2023](https://arxiv.org/abs/2303.05737) |
| Diarization | Confusion matrix | Nothing new | ADS_evaluation |
| Diarization | WDER, cpWER, speaker count error | Design pending | [AU-Harness](https://github.com/ServiceNow/AU-Harness) |
| Facts | Claim recall and precision against the reference note | A judge | [DocLens](https://github.com/yiqingxyq/DocLens) |
| Facts | Completeness, conciseness, and groundedness: segments judged entailed, partly, or not, against the reference note or the transcript | A judge | [Symphony](https://arxiv.org/abs/2610.08161) |
| Judge | Pairwise preference between two notes on eight PDSQI-9 dimensions, judged in both orders | A judge | [Symphony](https://arxiv.org/abs/2610.08161) |
| Attribution | The same check against the generated transcript: a fact absent from both was invented by the note step; present only in the generated one, it came from ASR | A judge | [Layered evaluation](https://www.medrxiv.org/content/10.64898/2026.09.22.26363642v1.full), preprint |
| Negation | Assertion status: present, absent, possible | Rules or a model | NegEx, i2b2 2010, CliniKnote labels |
| Medication | Drugs, doses, and coherence, with severity | Rules and a judge | TEME, AAMAS 2026 |
| Timestamps | Boundary error against reference times, within a tolerance | Predicted segments | DER collars, forced-alignment work |
| Judge | The 13 dimensions, in both modes | Validated local LLM | ADS_evaluation |
| Omissions | Facts the transcript establishes, graded by severity, each checked against the note; flag a note missing a critical one | A judge | [OmissionBench](https://arxiv.org/abs/2608.31016); MED-OMIT |
| Facts | Coverage, omission, contradiction against the reference note, then unmatched facts checked against the transcript | A judge | [Latif et al. 2026](https://proceedings.mlr.press/v333/latif26a.html) |
| Sections | Reference-based metrics per section, as ACI-BENCH scores its four divisions | `note_sections` headings | ACI-BENCH, CliniKnote |
| ASR severity | Medication, dose, and consistency errors graded none, minor, or major | A judge | TEME, AAMAS 2026 |

## Validating judges

A judge is a metric too, and PriMock57 can check it. Five clinicians each post-edited
the same 285 notes, 4 generated and 1 human per consultation, comparing them with the
audio. They marked:

- **Incorrect statements:** sentences with at least one factual error against the audio.
- **Omissions:** medical facts that belonged in the note and are missing.
- **Critical:** either kind, when it would create medico-legal liability.
- **Post-editing time.**

A judge's counts and spans can be compared with theirs, against the clinicians' mean,
since they disagree with each other. Expect judges to miss omissions: a 2026 preprint
found eight judge designs near chance on them. See [evaluation notes](evaluation_notes.md).

The judge's two modes fit `Requires`: one needs a `note` reference, the other a
`transcript`.

## Planned perturbations

A perturbation changes an input before the pipeline and creates a `variant`, in its own
run, scored against the clean run of the same samples. Two kinds, since they answer different
questions:

**Robustness.** The reference stays valid, so the question is how much each metric
drops against `clean`.

| Layer | Perturbation | Source |
| --- | --- | --- |
| Audio | Noise at a set SNR, cuts, telephone codec, speed | Common ASR practice |
| Transcript | Mask 10% of words, phrases, or sentences | EvalLLM4Health, ADS |
| Transcript | Swap speaker labels on 10% of turns | EvalLLM4Health, ADS |
| Transcript | Insert off-topic talk | EvalLLM4Health |
| Transcript | Remove a drug, number, diagnosis, or symptom mention | Latif et al. 2026 |
| Transcript | Flip a negation; swap a drug for a similar-sounding one | Latif et al. 2026 |

**Safety probes.** The input carries a trap, so each needs its own check of how the
note handled it: omitted, preserved, corrected, or flagged, as Latif et al. grade it.

| Probe | Expected | Source |
| --- | --- | --- |
| Impossible values, like BP 500/300 | Flagged | EvalLLM4Health, ADS |
| Profanity in the patient's speech | Dropped | EvalLLM4Health |
| Rare diseases, new drugs | Kept as said, not swapped for known ones | ADS |

What they found:

- ADS, on 40 real prenatal visits: masking critical words lowered note scores mildly;
  swapping 10% of speakers barely mattered. GPT-4o flagged impossible values in 2% of
  cases and copied them unchanged in 60%. New drug names often broke transcription.
- Latif et al., one error per PriMock57 transcript: redundancy recovered 38.6% of
  errors, because the fact was said again elsewhere. Negation flips and similar drug
  names propagated when it wasn't. The paper points to its code at
  [LLM-Notes-Evaluation](https://github.com/ehsanlatif/LLM-Notes-Evaluation).

Repeats come first: without the clean variation between generations, a variant's
drop means little. ADS generated 5 notes per transcript.
