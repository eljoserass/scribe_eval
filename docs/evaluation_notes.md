# Evaluation notes

What the literature says about measuring scribes, and what follows for this project.
Read in November 2026: mostly results sections and abstracts, not every paper in full.
ACI-BENCH and ADS details come partly from summaries of their full text.

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
- **PriMock57's generated notes came from ASR,** Google Speech-to-text, and clinicians
  judged them against the audio. Its error labels already mix ASR and note errors. [1]
- **Most serious errors may start in ASR.** Scoring notes from a perfect transcript and
  end to end, one preprint traced 87% of serious errors to ASR and 7% to the note step.
  Its data is synthetic, TTS in 77 languages; three LLM raters, not clinicians,
  labelled the errors; its code and data aren't public. ACI-BENCH saw little
  difference on n-gram metrics, which weight all facts equally. [10, 11]
- **Completeness separates systems most.** Corti's benchmark of three commercial
  scribes found groundedness at 90–98% for all and completeness at 63–77%. Its judge
  is GPT-5.4; three in-house clinicians endorsed its verdicts in 86% of 240 checks,
  ratifying them rather than judging independently. Authored by one of the vendors
  compared; no code released. [14]
- **Judges miss omissions.** Eight judge designs scored 0.79–0.94 on added or altered
  content and 0.50–0.63 on omissions. Extracting facts from the transcript first, then
  checking each one, recovered part of it. The same preprint found a material
  discrepancy with the transcript in all 53 PriMock57 clinician notes it audited,
  10.7 per note on average. Its benchmark (CC BY 4.0) and harness (MIT) are public. [12]
- **Flag missing critical facts; don't average coverage.** In the same preprint, mean
  coverage barely moved between clean notes and notes with an omission, 0.953 against
  0.929. Flagging any note missing a fact graded critical caught 24.6% of omissions at
  2.7% false alarms. Facts were extracted from the transcript without seeing the note,
  about 39 per note. [12]
- **Reference notes miss what the transcript supports.** Of generated content absent
  from PriMock57's notes, 92% was supported by the transcript. Across eight LLMs,
  omissions were 8.5–24.0% of facts and contradictions 6.2–7.9%. So unmatched facts
  need a second check against the transcript before they count as invented. [15]
- **Redundancy hides transcription errors.** With one error injected per transcript,
  38.6% were recovered because the fact came up again. Negation flips and
  similar-sounding drugs propagated when it didn't. [15]
- **ASR errors show up in judges, not in BERTScore.** Summaries from ASR transcripts
  matched BERTScore against references as well as those from human transcripts, but
  an LLM judge scored them 2 to 5 points lower. That judge, o1, correlated 0.816 with
  clinical experts on 6 criteria, positives and negatives among them. [17]
- **Lower WER, more dangerous errors.** In Spanish, Whisper beat a medical ASR on WER,
  3.3% against 4.7%, yet made 21 major clinical errors against 12, judged by three
  GPT-4o agents on medication, dose, and consistency. Its Clinical BERTScore gave 0.97
  to a drug name that doesn't exist. [16]
- **Judges agree least on form.** GPT-4o matched clinicians well on relevance,
  completeness, and comprehension, poorly on fluency, brevity, coherence, and
  structuring. Clinicians agreed with each other about 54% of the time. [2]
- **Sections need detection rules.** ACI-BENCH splits notes into four divisions with
  regular expressions; a short generated section can go undetected and score as
  empty. CliniKnote scores 15 sections, or 4 groups of them. [11]
- **Publish the facts behind the counts.** One open leaderboard scores omissions
  against a fact list it doesn't release, so its counts can't be checked.

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
10. [Mind the gap: emergent clinical risk at the interface of two individually safe AI systems](https://www.medrxiv.org/content/10.64898/2026.09.22.26363642v1.full), medRxiv 2026.
11. Yim et al., [ACI-BENCH](https://arxiv.org/abs/2306.02022), Scientific Data 2023.
12. [LLM Judges Verify Presence, Not Absence](https://arxiv.org/abs/2608.31016), 2026.
13. Xie et al., [DocLens](https://arxiv.org/abs/2311.09581), ACL 2024.
14. Varab et al., [Symphony for Text Generation: Benchmarking Clinical Note Generation](https://arxiv.org/abs/2610.08161), arXiv 2026.
15. Latif et al., [Evaluating Robustness of LLM-Based Ambient Scribes for SOAP Note Generation](https://proceedings.mlr.press/v333/latif26a.html), CHIL 2026, PMLR 333.
16. Villarroya-Martinez et al., [TEME: A Multi-Agent Evaluation Framework for Spanish Medical Speech Recognition](https://doi.org/10.65109/RHNT7438), AAMAS 2026, extended abstract.
17. Sanni et al., [Afrispeech-Dialog: A Benchmark Dataset for Spontaneous English Conversations in Healthcare and Beyond](https://aclanthology.org/2025.naacl-long.426/), NAACL 2025.
18. Schumacher et al., [Extrinsically-Focused Evaluation of Omissions in Medical Summarization](https://arxiv.org/abs/2311.08303), ML4H 2024. MED-OMIT.
19. Srivastav et al., [Open ASR Leaderboard](https://arxiv.org/abs/2510.06961), arXiv 2025. Code: [open_asr_leaderboard](https://github.com/huggingface/open_asr_leaderboard).
