# Datasets

Reviewed in November 2026. PriMock57 figures were measured from its repository;
TEME-v1 figures from an inspection of its ZIP in October 2026. The rest comes from
each dataset's page, schema, or paper. Check licenses at the source before publishing.

## Common format

`scribe-eval prepare` converts a dataset once. After that, a single loader reads them
all. Artifacts use Scribnic's JSON.

```text
data/primock57/
├── dataset.json         # name, version, language, license, source, splits
└── samples/day1_consultation01/
    ├── audio.wav        # PCM16, mono, 16 kHz
    ├── diarized_transcript.json
    ├── note.json
    └── metadata.json
```

## Status

Only Smoke and PriMock57 are integrated.

| Dataset | Use | Artifacts | Size | Language | License |
| --- | --- | --- | --- | --- | --- |
| Smoke | Test the plumbing | Audio, transcript, fixture note | 1 | es | CC BY 4.0 |
| [PriMock57](https://github.com/babylonhealth/primock57) | End-to-end evaluation | Audio, transcript, note, human evaluation | 57 | en | CC BY 4.0 |
| [TEME-v1](https://zenodo.org/records/17280661) | The Spanish option: ASR, reference-free notes | TTS audio, transcript | 90 | es | CC BY 4.0 |
| [ACI-BENCH](https://github.com/wyim/aci-bench) | Transcript to note, official splits | Transcript, note | 207 | en | CC BY 4.0 |
| [MTS-Dialog](https://github.com/abachaa/MTS-Dialog) | Note sections, metric correlation | Short dialogue, note section | 1,701 | en | CC BY 4.0 |
| [CliniKnote](https://arxiv.org/html/2408.14568v2) | Long notes, negation | Transcript, note, entities | 1,200; 20 public | en | Unconfirmed |
| [Afrispeech-Dialog](https://huggingface.co/datasets/intronhealth/afrispeech-dialog) | ASR with accents | Audio, transcript | 49 per its table; 20 medical | en | CC BY-NC-SA 4.0 |
| [MedSynth](https://huggingface.co/datasets/Ahmad0067/MedSynth) | Text at volume | Dialogue, SOAP note | 10,035 | en | Unconfirmed |

**The main gap:** nothing here combines Spanish, real speech, and clinical notes.
TEME-v1 is synthetic and has no notes. Note quality in the intended clinical setting
can't be claimed without local consultations and reviewed notes.

## Smoke

One TEME-v1 dialogue with its audio: 33 turns, about 5 minutes. Its reference note is
a handwritten fixture, flagged in metadata. It tests code, not quality.

## PriMock57

Mock primary care consultations: 7 clinicians, 57 staff acting as patients.

| Measure | Value |
| --- | ---: |
| Total audio | 8.6 h |
| Median duration | 9.3 min (3.8–14.3) |
| Median transcript | 8,002 characters, 1,536 words |
| Longest transcript | 14,996 characters |
| Median utterances | 124 |
| Overlapping speech, approximate | 5.8% |

The converter:

- Downloads from a pinned commit, verifying audio against Git LFS pointers. `git-lfs`
  isn't needed.
- Averages the doctor and patient channels into one WAV, as `sox -m` does.
- Merges the two TextGrids by start time into a `DiarizedTranscript`.
- Strips `<UNSURE>` tags, keeping their words, and the `<UNIN/>` and
  `<INAUDIBLE_SPEECH/>` markers: 1,132, 1,359, and 814 of them. The last one is missing
  from the dataset's documentation.
- Uses a single `all` split, since there are no official ones.

Limits:

- **Notes aren't SOAP.** They're free text in clinical shorthand, like
  `3/7 hx of diarrhea`. Lexical metrics will punish style as well as content.
- **Input limit.** Scribnic rejects more than 12,000 characters. On clean transcripts
  that's 3 of 57 consultations, roughly 7 with speaker labels. These count as failures.
- **Whisper timestamps.** Scribnic rejects a segment ending more than 0.5 s past the
  audio. Whisper's timestamps can overshoot that: `day1_consultation03` ends at 545.0 s
  on 544.32 s of audio, so the whole transcript is lost. Open until Scribnic handles it.

`human_eval_data` is the hidden gem: 5 evaluators on notes from 10 models, with
post-editing time and incorrect statements and omissions marked critical or not. It
can show whether a metric tracks human judgment, without running a model.

## TEME-v1

90 Spanish dialogues across 10 specialties, 30 of them cardiology. 2,330 turns, two
speakers each, 3.96 h, about 2 min 38 s on average. A 94 MB ZIP.

- **Audio is TTS.** WER will flatter a system: no noise, overlap, or spontaneous speech.
- **No notes.** `context` is a short summary, not a note, and must not reach the prompt.
- **Estimated timings.** `duration_estimate` isn't gold for DER.
- **Conversion** still needs MP3 to WAV.
- [TEME, AAMAS 2026](https://doi.org/10.65109/RHNT7438) is a related framework scoring
  medication, dose, and coherence with judges and severity rules. Unread so far.

## ACI-BENCH

Full conversations and notes with MEDIQA splits: train 67, valid 20, three tests of
40. Notes also come by section. No audio in the repository. Some subsets keep swapped
speaker tags on purpose, so speakers aren't a perfect reference.

## MTS-Dialog

Train 1,201, validation 100, two tests of 200. Use the original repository; the
Hugging Face mirror has only part of it, under a different stated license.

- Each row is a short dialogue and one note section, like `GENHX` or `ALLERGY`. Good
  for sections, not for a whole SOAP note.
- `Correlation-Study` has manual scores for validating metrics.
- `Augmented-Data` includes back-translation through Spanish. Not Spanish clinical data.

## CliniKnote

1,200 dialogues role-played by clinical staff, with K-SOAP notes: SOAP plus a keyword
section. Averages 59 turns and 954 words per dialogue, much closer to a real visit
than MTS-Dialog.

- **Availability.** The full corpus doesn't appear to be public. The
  [paper's demo](https://github.com/catalwaysright/K-SOAP-GOODLABS) has 20 dialogues,
  20 notes, and 200 entity segments, with no stated license.
- **Why it matters.** PRESENT, ABSENT, FAMILY, and UNKNOWN labels are what negation
  evaluation needs.
- Its paper also points to NoteChat, and to BERTScore, BLEURT, and reference-free
  QuestEval.

## Afrispeech-Dialog

[NAACL 2025](https://aclanthology.org/2025.naacl-long.426.pdf). African-accented
English at 48 kHz. No note column, though the paper mentions expert summaries. Only
9 of 20 medical transcripts have timestamps. The abstract says 50 conversations; the
table adds up to 49. Non-commercial license.

## MedSynth

[arXiv 2508.01401](https://arxiv.org/html/2508.01401v2). Dialogues and SOAP notes
generated with GPT-4o. Useful for load and volume, not as clinical evidence.

## Not usable

Wang et al. use 40 real consultations that aren't public. Their repositories offer
code and metrics, not data.

## Unread

The [StatPearls SOAP guide](https://www.ncbi.nlm.nih.gov/books/NBK482263/) blocked
automated access, so its criteria aren't reflected here.
