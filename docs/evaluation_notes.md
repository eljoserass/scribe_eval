# Evaluation notes

What the literature says about measuring scribes, and what follows for this project.
Read in November 2026: mostly results sections and abstracts, not every paper in full.

## Findings

- **Clinician notes aren't ground truth.** In PriMock57's human evaluation they held
  over 1 incorrect statement and almost 4 omissions each, part error and part style.
  Compare notes against the conversation, not only against another note. [1]
- **Beat note length first.** Against post-editing time, note length alone reaches 0.41.
  ROUGE-L gets 0.38, BERTScore 0.50, Levenshtein 0.55. [1]
- **Clinicians disagree.** Two clinicians correlate 0.29 on critical error counts in
  PriMock57 (our calculation). Wang et al. report about 54% agreement, and another study
  κ 0.505. That caps how well any metric can score. [1, 2, 6]
- **Average clinicians before correlating.** Against the mean of 5 clinicians per note,
  BERTScore reaches -0.59 on critical errors. Against single evaluations, -0.39 (our
  calculation).
- **Report errors by type.** Omissions were 83.8% of errors across 7 commercial scribes.
  Hallucinations are rarer but more often serious. Among major hallucinations, 43% were
  fabrications and 30% negations, most often in the Plan. [3, 4]
- **Negations are a large class.** About 43% of PriMock57's critical incorrect statements
  mention a negation (a crude keyword count, our calculation).
- **WER misses clinical risk.** Kendall's τ with clinician-labelled impact is about 0.2.
  An LLM judge reached κ 0.82, on 50 cases, with a remote model. [6]
- **Term-restricted WER is deterministic,** with no model at evaluation time. Its fixed
  list covers about 20% of annotated entities. [5]
- **Judges help but aren't clinicians.** Validate them first. [7]

## Checks in this project

| Check | Result |
| --- | --- |
| `levenshtein` and `bleu` against published scores | 285 of 285 identical |
| Impact of ASR errors, Kendall's τ (172 utterances, [6]) | WER 0.16, term errors 0.29 |
| Finding significant errors by term errors | Precision 67%, recall 25% |
| Finding significant errors by WER above 0.3 | Precision 35%, recall 73% |

Term errors are precise but miss most serious errors: lost negations like "No.", and
terms outside the list. "Paracetamol" isn't on it.

## Approach

Add reproducible, well-known metrics first, several at once, because each one fails in
its own way. Wang et al.'s framework is the reference for what to cover: see the coverage
table in [metrics](metrics.md). Judges come later, with the model swappable.

## What to use a metric for

A metric earns trust by passing four checks: it tracks the clinicians' mean, it beats
note length, it finds the specific errors clinicians marked, and it catches errors
injected on purpose. PriMock57's `human_eval_data` and the labels in [6] support all
four without running a scribe.

## Sources

1. Moramarco et al., [Human Evaluation and Correlation with Automatic Metrics in Consultation Note Generation](https://arxiv.org/abs/2204.00447), ACL 2022.
2. Wang et al., [An evaluation framework for ambient digital scribing tools](https://doi.org/10.1038/s41746-025-01622-1), npj Digital Medicine 2025. Code: [ADS_evaluation](https://github.com/AutoEvaluation/ADS_evaluation).
3. Asgari et al., [A framework to assess clinical safety and hallucination rates](https://doi.org/10.1038/s41746-025-01670-7), npj Digital Medicine 2025.
4. [Clinical AI Scribes in primary care](https://doi.org/10.1136/bmjdhai-2025-000092), BMJ Digital Health & AI 2025.
5. [MedWER](https://arxiv.org/abs/2609.05728), 2026. Code: [medwer](https://github.com/Nordis-Tech/medwer).
6. [WER is Unaware](https://arxiv.org/abs/2511.16544), 2025. Data: [wer-is-unaware](https://github.com/Ufonia/wer-is-unaware).
7. [When Reasoning Hurts](https://arxiv.org/abs/2605.24902), 2026.
8. [An Investigation of Evaluation Metrics for Automated Medical Note Generation](https://arxiv.org/abs/2305.17364), 2023.
9. [Validating Digital Scribes: A Scoping Review](https://doi.org/10.1007/s10916-026-02392-3), Journal of Medical Systems 2026.
