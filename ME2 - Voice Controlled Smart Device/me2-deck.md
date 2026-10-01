# ME2 — Voice Controlled Smart Device

*Always-on keyword + intent recognition at the edge · no cloud round-trip*

2 October 2026 · Rowel Atienza (with OnIt, AI agent) · Model · Dataset · Training on CPU · Validation on RPi 4

> **Architecture revision (vs. the original template).** The original deck assumed a
> CTC/attention causal encoder trained on an A100 with KV-cache streaming. The built
> system (pi test v6) is instead a **classic HTK/PocketSphinx-style HMM/GMM
> recognizer, trained from scratch by bootstrap EM on CPU** — no pre-trained acoustic
> model, no GPU, no attention. Sections below reflect the actual v6 pipeline.

## Model

| Item | Value |
|---|---|
| Architecture | HMM/GMM: 37 phones × 3-state left-to-right HMM, 4-component diagonal GMM per state, tied across all words |
| Parameters / weights | **35.4 k params · 0.14 MB** (f32) |
| Vocabulary | 109 words (88 in-scope + number words), 37 phones (CMU ARPAbet subset) |
| Language model | 93 phrase variations → 19 command classes (grammar FSA) + free unigram |
| Rejection | two-level Viterbi (constrained vs free) with score-gap REJECT — unknown input is rejected, not force-mapped |
| Export | ONNX (front-end + GMM emissions) — pending, after training completes |

## Validation on the Raspberry Pi 4

| Item | Value |
|---|---|
| Keyword / intent acc | ⟨TBD — eval pending, training in progress⟩ % / ⟨TBD⟩ % |
| False-accept rate | ⟨TBD⟩ % |
| Latency p95 / RTF | ⟨TBD⟩ ms / ⟨TBD⟩ |
| Runtime | onnxruntime (post-training export) · 1 thr (numpy reference until then) |

Reference (previous Mark dataset, 171 clips): PocketSphinx en-us 9.6 MB → 78.4 % cmd / 83.0 % intent;
trained 0.8 MB AM → 84.8 % / 88.3 %. New-dataset comparison pending.

## Dataset

| Item | Value |
|---|---|
| Source | HuggingFace `airimonda/ai231-me2-voice-commands` — real voice + synthetic + CommonVoice_en / FluentSpeechCommands / SLURP / SNIPS; numerals: MLEnd + SpeechCommands v2 |
| Hours / utts | **30.9 h / 81,686** (train 10,682 · test 4,418 · holdout 196 · numerals 66,390) |
| Speakers | 315 (train) · 121 (test, **zero overlap with train**) · 2,547 (numerals) |
| Labels | **19 intents** + OUT_OF_SCOPE · **18 slot values** (times, percents, temperatures, reminder texts) |

## Training (revised: CPU, bootstrap EM — no A100)

| Item | Value |
|---|---|
| Cluster | 256-core CPU (sandbox), 32 workers for feature extraction |
| Objective | maximum likelihood — GMM EM + L2R-masked transition re-estimation (no gradients) |
| Optimiser | bootstrap EM (HTK-style): proportional-segmentation init → 8 rounds of soft (forward-backward) alignment |
| Steps / loss | 8 rounds / negative log-likelihood (round 0: 38.5 M effective phone-frames, 36/37 phones active) |
| Training set | 28,834 clips (8,834 in-scope train + 20,000 numerals) |

## Pipeline (revised diagram)

```
Mic · 16 kHz
  → cepstral front-end: pre-emphasis 0.97 · Hamming 25.6 ms / 10 ms hop · 512-FFT ·
    26 mel filters (133–6855 Hz) → 13 cepstra (orthonormal DCT-II) · CMN · +Δ/+Δ²
    → 39-dim / frame
  → GMM emissions: 37 phones × 3 states, frame-by-frame (inherently streaming — no KV cache)
  → Viterbi over grammar FSA (93 phrases) + free unigram
  → intent head: 19 command classes + score-gap REJECT
  → slot head: numerals / times / percents / temperatures from the decoded phrase
  → actuator
Raspberry Pi 4
```

## To Be Submitted — what the reviewer will look for

| # | Item | Status |
|---|---|---|
| 1 | Repo public, one-command reproduction | in progress — repo public; `python3 train_am.py && python3 eval_v6.py` |
| 2 | Dataset licensed and citable (DOI) | todo — HF card has no licence set yet |
| 3 | Training logs + final checkpoint committed | in progress — checkpoint + eval report to be committed after EM completes |
| 4 | Pi 4 latency reproduced by the posted script | pending |
| 5 | Held-out test set, unseen speakers | **done** — 4,418 clips, 121 speakers, zero overlap with train |
| 6 | Baseline of comparable size compared | in progress — PocketSphinx en-us (9.6 MB) as baseline; new-test-set comparison pending |

- GitHub repository: `raesoriano/Machine-Learning-Operations` (public) — `ME2 - Voice Controlled Smart Device/pi test v6/`
- Dataset location: HuggingFace `airimonda/ai231-me2-voice-commands` (licence + DOI pending)
- Training hardware: 256-core CPU, wall-clock ≈ 2 h, seed 0 *(revised from A100)*
- Model weights: to be committed to the repo after training (0.14 MB)
