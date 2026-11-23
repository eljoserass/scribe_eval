# Roadmap

Draft. Each release is a GitHub milestone, and the milestone's issues are the source of
truth for what goes in. This page says what each release is for and lists candidates
not yet scheduled. Versions stay at 0.x while interfaces move; `CHANGELOG.md` records
what shipped.

## 0.1.0: Scribnic runs PriMock57

A scribe that can't get through a dataset can't be evaluated, so the first release is
one complete run.

- Install on Linux (WSL2 x86_64) and macOS (Apple Silicon), and prepare PriMock57.
- Run Scribnic's English workflow on all 57 consultations with `--no-score`, then score
  them with every current metric.
- `scribe-eval runs` and `compare` work on that run.
- CI, license, contributor files, and a release process. Release notes state the tested
  platforms.
- NegBLEURT. `medical_term_wer` removed.

## 0.2.0: other scribes, and metrics with medical knowledge

Pipelines come first: the harness claims to be scribe-agnostic, and a second and third
scribe test that claim.

- Berta AI Scribe and OpenScribe run PriMock57 through the same experiments, and
  `compare` puts them beside Scribnic.
- Data-parallel runs: split a run's samples into shards, run each shard as its own
  isolated pipeline instance, and score or compare the shards as one run. Check first
  that several Scribnic instances can run side by side.
- Streaming against batch on the same audio.
- ACI-BENCH, scored as MEDIQA-Chat 2023 scores it, so results match published numbers.
- MEDCON and affirmed and negated concept F1.
- First judge runs on PriMock57.

## Later

Themes without a release yet. Each has issues labelled `backlog`.

- **Judges and their validation:** the ADS judge, judge evaluation against PriMock57's
  human data, OmissionBench omissions, Latif et al.'s fact protocol, facts as a
  dataset reference.
- **Errors weighted by clinical severity.** Mistaking paracetamol for omeprazole
  matters more than a cousin for a brother. Overlap and embedding metrics weigh both
  alike; clinical BERTScore and MedBERTScore weigh every medical term alike. The
  literature grades severity per fact, by clinicians or a judge: major if it would
  change diagnosis or management (Asgari et al. 2025), or none, minor, and major for
  drugs and doses (TEME).
- **Own data:** a collection protocol and splits, and Spanish clinical references.
- **Robustness:** perturbations as independent runs.
- **Infrastructure:** experiment tracking.
- **Documentation:** reorganize `docs/` by [Diátaxis](https://diataxis.fr/): tutorials,
  how-to guides, reference, and explanation. Today the pages mix all four.
- **Metric organization.** Model-based metrics keep growing: BERTScore, BLEURT,
  NegBLEURT, Clinical BERTScore, and the candidates below. Revisit how they are grouped,
  configured, and how their models are loaded and recorded, before adding many more.

## Candidate metrics

Not scheduled. "Have" is what the harness computes today.

| Candidate | Have | Would add | Source |
| --- | --- | --- | --- |
| ROUGE-Lsum | ROUGE-1, 2, L | The summary-level variant used by ACI-BENCH and CliniKnote | MEDIQA-Chat 2023 |
| MedBERTScore, MedBARTScore | BERTScore | Extra weight on medical terms. MedBARTScore led on factual precision and hallucination for MTS-Dialog | Ben Abacha et al. 2023; code without license |
| MIST, MIST-Comb | ROUGE-1, BERTScore, BLEURT | UMLS knowledge-graph embeddings, alone or averaged with those three. MIST-Comb tied for the best overall correlation | Ben Abacha et al. 2023; MIST code not released |
| ClinicalBLEURT | BLEURT | BLEURT fine-tuned on clinical notes. Best on hallucination in three in-house datasets | Ben Abacha et al. 2023; checkpoint availability unverified |
| SapBERT score, CUI F-score | Nothing | UMLS concept embeddings and CUI overlap. SapBERT score was the only one of ten metrics with a significant correlation, still weak (Pearson 0.17) | Croxford et al., AMIA 2024 |
| QuestEval | Nothing | Reference-free, question-answering consistency | Scialom et al. 2021; used by CliniKnote |
| NegBLEURT by sentence | NegBLEURT on whole notes | Pairs of matched sentences, where the paper's scale holds | Anschütz et al., INLG 2023 |
| Metric validation | Nothing | Correlate metrics with MTS-Dialog's correlation-study annotations and PriMock57's human evaluation | Ben Abacha et al. 2023; Moramarco et al. 2022 |
| Clinical BERTScore validation | Clinical BERTScore | Agreement with the paper's clinician preferences | Shor et al. 2023 |

CliniKnote's contribution is a way to generate notes, keywords first, then K-SOAP. That
belongs in a pipeline, not here.

## Before 1.0

Not dated. 1.0 means:

- The `Pipeline` protocol, prepared dataset format, prediction rows, and telemetry
  contract are stable, with migration notes when they change.
- At least two pipelines besides fakes run the same experiments.
- The main metrics are validated against human annotations, with the results in the
  docs.
