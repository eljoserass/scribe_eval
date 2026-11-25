# Metrics

Each metric declares what it needs and how it aggregates. Experiments can name a
metric or a whole group. Nothing here measures clinical safety yet.

## Available

| Group | Metric | Predicted | Reference | Aggregates |
| --- | --- | --- | --- | --- |
| `transcription` | `wer`, `cer` | `transcript` | `transcript` | Global, sample mean |
| `transcription` | `transcript_rouge_1`, `transcript_rouge_l` | `transcript` | `transcript` | Mean |
| `transcription` | `transcript_bleu` | `transcript` | `transcript` | Corpus, mean |
| `transcription_semantic` | `clinical_bertscore` | `transcript` | `transcript` | Global, sample mean |
| `diarization` | `der` | `diarized_transcript` | `diarized_transcript` | Global, by cause |
| `diarization` | `jer` | `diarized_transcript` | `diarized_transcript` | Mean |
| `note_format` | `note_words` | `note` | — | Mean, max |
| `note_format` | `flesch_reading_ease` | `note` | — | Mean |
| `note_format` | `compression_ratio` | `note` | `transcript` | Global |
| `note_reference` | `rouge_1`, `rouge_2`, `rouge_l` | `note` | `note` | Mean |
| `note_reference` | `bleu` | `note` | `note` | Corpus, mean |
| `note_reference` | `meteor` | `note` | `note` | Mean |
| `note_reference` | `levenshtein` | `note` | `note` | Mean, per character |
| `note_semantic` | `bertscore` | `note` | `note` | Mean, with precision and recall |
| `note_semantic` | `bleurt` | `note` | `note` | Mean; precision and recall with `sentences` |
| `note_semantic` | `negbleurt` | `note` | `note` | Mean; precision and recall with `sentences` |
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
| Structuring | Structural precision/recall | Skipped: neither the paper nor its code defines it. Their judge rates structuring, planned below |
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
- **`resources`** read the [telemetry contract](architecture.md#telemetry), so they
  work for any pipeline that fills it in. Only `elapsed_seconds` and `real_time_factor`
  work for every pipeline, because the harness measures them. The rest are `n/a` when
  a pipeline doesn't report them. Memory is sampled RSS of the pipeline's processes,
  not the whole machine, and may double count shared pages. For Scribnic that includes
  the client, about 0.12 GB. Model loading is in `startup.jsonl`, not here.
- **Peak memory and order.** Unless `isolate_samples = true`, a sample inherits what
  earlier ones left, so peaks mix consultation length with order. The report's RAM
  start and end columns show what stayed.

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

`bertscore`, `bleurt`, `negbleurt`, and `clinical_bertscore` run neural models, so they
need the `semantic` extra: `uv sync --extra semantic`. Weights download from Hugging
Face the first time a run scores with them, into its cache. Set `HF_HOME` to move it,
for instance to shared storage on a cluster, and `HF_HUB_OFFLINE=1` on nodes without
internet once it's filled. `device = "auto"` picks CUDA, then Apple MPS, then CPU.

Defaults reproduce each paper; `[metric_settings]` changes them per experiment, and
`manifest.json` records the effective model, layer, long-text strategy, and package
version.

```toml
[metric_settings.bertscore]
model = "microsoft/deberta-xlarge-mnli"  # default: roberta-large in English
device = "cuda"
long_text = "window"                       # default: truncate; see Long notes

[metric_settings.bleurt]
checkpoint = "lucadiliello/BLEURT-20-D12"  # default: BLEURT-20, about 2 GB
batch_size = 32                            # pairs per forward pass; default 16

[metric_settings.negbleurt]
checkpoint = "/shared/models/NegBLEURT"    # default: tum-nlp/NegBLEURT, 17 MB
long_text = "sentences"

[metric_settings.clinical_bertscore]
medical_weight = 0.4                       # the paper's tuned k
terms = "~/terms/medical_terms_es.txt"     # one term per line
```

| Metric | Default | Notes |
| --- | --- | --- |
| `bertscore` | `roberta-large`, layer 17, in English; multilingual BERT otherwise | [Zhang et al. 2020](https://arxiv.org/abs/1904.09675), through their `bert-score` package. F1, no IDF, no rescaling: what ADS_evaluation gets from Hugging Face `evaluate` with `lang="en"` |
| `bleurt` | BLEURT-20 | [Sellam et al. 2020](https://arxiv.org/abs/2004.04696). Google's code needs TensorFlow, so this uses the `bleurt-pytorch` port and its conversions of the official checkpoints. Not comparable across checkpoints |
| `negbleurt` | `tum-nlp/NegBLEURT`, English only | [Anschütz et al. 2023](https://aclanthology.org/2023.inlg-main.12/). The authors' checkpoint, loaded with `transformers` as their [model card](https://huggingface.co/tum-nlp/NegBLEURT) and [code](https://github.com/MiriUll/negation_aware_evaluation) do. Weights CC BY-SA 4.0. Adapted below |
| `clinical_bertscore` | BERTScore's model, k = 0.4, MedWER's term list | [Shor et al. 2023](https://arxiv.org/abs/2303.05737), adapted below |

By default, inputs past 512 tokens are truncated, and `truncated` counts them.
Generated notes can reach that, and [Long notes](#long-notes) lists the
alternatives. Transcripts always would, hence Clinical BERTScore's split below.

**NegBLEURT** is BLEURT-tiny fine-tuned to score a negated statement low. The authors
trained it on CANNOT-WMT: 62K English sentence pairs, 88% from their CANNOT dataset,
labeled 0 when one sentence negates the other and 1 when it paraphrases it, and 12%
WMT ratings. Its output is unbounded but mostly falls between 0 and 1. The reference
goes first in the pair, as in their code.

- **Whole notes by default.** The paper scored sentence pairs. By default a note and
  its reference are one pair, so the paper's scale doesn't carry over. On a synthetic
  12-sentence SOAP note, 261 tokens as a pair, the identical note scored 0.62,
  flipping "denies chest pain" to "reports chest pain" 0.47, and dropping that
  sentence 0.39. One case, but enough to not read a whole-note score as a negation
  detector. Compare it between runs, not with the paper. `long_text = "sentences"`
  scores sentence pairs instead; see [Long notes](#long-notes), where it doesn't
  separate a flip from a drop either.
- **Truncation.** Pairs past 512 tokens, the length BLEURT-tiny was trained with, are
  cut. The longer text loses tokens first, as in BLEURT. The checkpoint's tokenizer
  sets no limit, so `max_length` does, for each sentence pair too.
- **English only,** like the model. Other languages are `n/a`, with the reason, and
  need no extra.

Checked through this metric, on CPU: against "Ray Charles is legendary.", the model
card's example scores 0.8409 for "Ray Charles is a legend." and 0.2601 for "Ray Charles
isn't legendary.", as the card reports. The paper's Figure 1 shows 0.84 and 0.27. Its
CANNOT-WMT and DEMETR results weren't reproduced.

**Clinical BERTScore** is `k × BERTScore_medical + (1 − k) × BERTScore_all`, where the
medical part weighs only tokens of medical terms and numbers. The paper released no
code, so three things differ:

- Its 20K medical words come from Google's Knowledge Graph and aren't public. The
  19K-term list from [MedWER](https://github.com/Nordis-Tech/medwer), a preprint,
  stands in, so English only unless `terms` points to another list.
- It scored single sentences. Here each reference utterance is paired with the
  predicted words aligned to it, as for WER, and the sample weighs utterances by
  reference words.
- Text is normalized as for WER. Numbers count as medical, as digits or spelled out.

For Spanish, TEME's authors used `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`
with k = 0.4; `model` and `terms` reproduce that setup given a term list.

The paper's clinician preferences (CTP) are public, so the adaptation can be checked
against its reported agreement. Not done yet.

The `bertscore_global` detail is plain BERTScore over the same utterance pairs.

## Long notes

The default models read 512 tokens: BERTScore per text, BLEURT and NegBLEURT per
pair. On a 53-note PriMock57 run, BLEURT and NegBLEURT cut 7 pairs and BERTScore up
to 6, and a 2,674-word note stuck in a loop was scored on its opening only.
`long_text` in `[metric_settings.<metric>]` sets what happens past the limit. Each
metric takes only the strategies that suit its model; another value fails when the
experiment loads.

| `long_text` | Metrics | What it does | From |
| --- | --- | --- | --- |
| `truncate` | `bertscore`, `bleurt`, `negbleurt` | Scores what fits; `truncated` counts cut pairs. The default, so earlier results stay comparable | — |
| `window` | `bertscore` | Embeds a text past the limit in 512-token windows overlapping by 100 tokens, then matches tokens over the whole text. Pairs that fit score as with `truncate` | MedBERTScore-SP: [Ben Abacha et al., Findings of ACL 2023](https://aclanthology.org/2023.findings-acl.161/) |
| `sentences` | `bleurt`, `negbleurt` | Scores every reference sentence against every generated one. Precision averages each generated sentence's best match, recall each reference sentence's; the value is their F1 | SMART-1: [Amplayo et al., ICLR 2023](https://arxiv.org/abs/2208.01030) |
| `chunks` | `bleurt` | Scores each generated sentence against reference chunks of about 350 words, keeps its best chunk, and averages. AlignScore's aggregation with BLEURT as the scorer: an adaptation | [Zha et al., ACL 2023](https://arxiv.org/abs/2305.16739) |
| `sections` | — | Pending, below | ACI-BENCH, MEDIQA-Chat |

`window` changes only pairs past the limit. `sentences` and `chunks` change every
note's score, and `chunks` measures precision only, so neither compares with
`truncate`. `manifest.json` records the strategy and its parameters, and
`scribe-eval compare` warns when runs used different ones.

Aggregates keep `truncated`. It's 0 with `window`; with `sentences` and `chunks` it
counts notes where a sentence or chunk pair still passed the limit. `windowed` counts
BERTScore pairs that needed windows, and `pairs` totals the pairs scored. Each pair is
a model call: 74 generated sentences against 74 reference ones make 5,476.

**Window.** It follows the paper: segments of at most 512 tokens, overlapping by 100,
embedded independently and concatenated, then BERTScore's greedy matching over the
whole text. The [authors' code](https://github.com/abachaa/EvaluationMetrics-ACL23)
has no license, so it was read, not copied. Differences from it:

- Their code drops the 100 overlapping tokens from each later window before embedding
  it, so each window holds 412 new tokens with no shared context, and only the text's
  ends carry special tokens. Here each window is a complete input, the overlap is read
  as context, and a token keeps the embedding from the first window that holds it.
- Their code windows every text in a batch once the padded batch passes 512 tokens,
  so a short text batched with a long one is split too. Here only a text past the
  limit is.
- Their code uses `deberta-xlarge-mnli` whatever model is asked for. Here it's the
  metric's model, `roberta-large` by default.
- MedBERTScore also gives medical terms, found with MedCAT, more weight. Not here.

In the paper's tables, MedBERTScore-SP correlates with human judgments within 0.01 of
MedBERTScore-P on every dataset, so windows read past 512 tokens without a shown gain.
Checked on CPU with `roberta-large`: on a pair that fits, the windowed path matches
`bert-score` to 1.2e-7.

**Sentences.** SMART scores texts with sentences as units and any sentence-pair
scorer, BLEURT among those it tested. As in
[its code](https://github.com/google-research/google-research/tree/master/smart_eval)
for BLEURT, one matrix of scores, reference sentence first, serves both directions.
Its precision is also SummaC-ZS's max-then-mean
([Laban et al., TACL 2022](https://arxiv.org/abs/2111.09525)). Differences:

- SMART also matches against the source document and keeps the higher score. Here
  only the reference note counts.
- Notes are split at line breaks, then with NLTK's Punkt. Headings and list items
  lack final punctuation, and Punkt alone, as SMART uses it, would merge them.
- Only SMART-1, not SMART-2 or SMART-L.

For NegBLEURT, sentence pairs are what it was trained on.

**Chunks.** AlignScore splits its context into chunks of whole sentences:
`words // 350 + 1` of them, with the same number of sentences each. Its code counts
words on whitespace, where the paper says tokens; this follows the code. Each claim
sentence keeps its best chunk, and the claims average. Here the reference is the
context and the generated note the claims, so a note that leaves out half the
reference can still score high. A chunk and a sentence can still pass 512 tokens,
and `truncated` counts those notes. NegBLEURT doesn't take `chunks`: in the check
below it scored chunk pairs around −0.35, below its usual 0 to 1, for faithful and
wrong notes alike.

**Clinical BERTScore** takes no `long_text`. It already scores each reference
utterance with the predicted words aligned to it, so windows add nothing. It doesn't
count truncation; an utterance pair past 512 tokens would be cut, which hasn't been
checked.

**Sections.** ACI-BENCH and MEDIQA-Chat score each division of a note on its own.
That needs section detection (#10, #42), which doesn't exist yet. With it, `sections`
would be one more `long_text` value for all three metrics.

**Repetition** isn't penalized by design: each piece is matched to its best
counterpart, so a repeated sentence can match again. In the check below, a looping
note raised NegBLEURT's precision and lowered windowed BERTScore's, so neither is a
repetition measure.

**Other languages.**

- Sentences are split with Punkt's model for the dataset's language. NLTK has 19,
  Spanish among them; another language fails validation with `sentences` or `chunks`.
  The models download on first use into `nltk_data`; `NLTK_DATA` moves them.
- `window` works with any model BERTScore loads, such as multilingual BERT, its
  default outside English. BLEURT-20 builds on multilingual RemBERT. NegBLEURT is
  English only.
- Unverified: Punkt on Spanish notes, and every strategy outside English. None has
  been compared with human judgments in any language.

**Check.** A synthetic 502-word English note, 745 RoBERTa tokens, with each section on
its own lines, against four versions of it, on CPU:

- *generated*: five sentences paraphrased.
- *cut*: its first six sections, 198 words.
- *looping*: *cut* followed by the medication list 40 more times, 1,278 words.
- *tail errors*: *generated* with four errors after token 595: a tenth of the dose,
  amoxicillin despite a penicillin allergy, doubled metformin, and no 999 call for
  chest pain.

| Metric | `long_text` | generated | cut | looping | tail errors |
| --- | --- | --- | --- | --- | --- |
| `bertscore` | `truncate` | 0.987 | 0.947 | 0.910 | 0.987 |
| `bertscore` | `window` | 0.988 | 0.934 | 0.865 | 0.986 |
| `bleurt` | `truncate` | 0.801 | 0.801 | 0.801 | 0.801 |
| `negbleurt` | `truncate` | 0.283 | 0.283 | 0.340 | 0.283 |
| `negbleurt` | `sentences` | 0.859 | 0.676 | 0.688 | 0.843 |

- With `truncate`, BLEURT scored all four the same: a pair keeps about half its 512
  tokens from each note, and all four open alike. No metric saw the tail errors with
  `truncate`.
- Split notes tell *cut* and *looping* from *generated* through recall: 0.56 against
  0.86 for NegBLEURT. *Looping* raised its precision, 0.90 against 0.86.
- The tail errors cost NegBLEURT 0.017 and windowed BERTScore 0.002. Four wrong
  sentences in 74 barely move a mean; none of these detects errors.

NegBLEURT on another synthetic note, 12 sentences: identical, "denies chest pain"
flipped to "reports chest pain", and that sentence dropped.

| `long_text` | identical | flipped | dropped |
| --- | --- | --- | --- |
| `truncate` | 0.634 | 0.474 | 0.530 |
| `sentences` | 0.920 | 0.897 | 0.897 |

With `sentences`, the flipped sentence's best match was another reference sentence,
not the one it negates, so a flip cost what a missing sentence did. One case each.

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
| Sections | Reference-based metrics per section, as ACI-BENCH scores its four divisions; also `long_text = "sections"` for the model metrics | Section detection, as ACI-BENCH does with regular expressions | ACI-BENCH, CliniKnote |
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
