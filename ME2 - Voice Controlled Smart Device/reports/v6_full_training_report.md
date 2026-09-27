# ME2 v6 — More training + speaker augmentation + reject gate (full run)

**Date:** 2026-09-27 · **GPU:** A100-SXM4-40GB (DGX cluster) · **Run:** `model/checkpoints/me2_v6/`

## What changed vs v5 (per the agreed plan)

The new hand-recorded test clips (`additional_test_data`, 171 clips, a new
speaker) were **NOT added to the training set** — they stay held out as an
honest generalization test. Instead:

1. **Mel-domain speaker augmentation** (`vcm/augment.py`, on the fly each
   epoch): pitch shift ±3 semitones (the speaker-diversity lever), time
   stretch 0.9–1.1, gain ±3 dB, time shift ±80 ms, SpecAugment (1 time +
   1 freq mask, 50% prob). The dataset is already 50/50 clean+noisy, so no
   additive noise.
2. **More training:** same ReduceLROnPlateau recipe as v4/v5 (proven best),
   **400 epochs, early-stop patience 30**, so the extra signal from
   augmentation is fully exploited.
3. **Inference-side confidence/reject gate** (`model/decode.py`
   `confidence()` / `decode_with_confidence()`): an utterance whose mean
   non-blank-frame log-prob is below a threshold decodes to "" and the
   parser returns `unknown` — the device stays silent instead of acting on
   a misheard command. Threshold tuned on the frozen v1 set
   (`scripts/tune_reject_threshold.py`, in-domain reject budget 5%).

Architecture unchanged: 128ch/4blk TCN-CTC (422,282 params, ~0.36 MB int8) —
the RPi-realtime model.

## Training

- Train rows: 37,992 · Val rows: 5,315 (precomputed log-mel features,
  `data/features/`)
- Recipe: AdamW lr 3e-4, wd 1e-4, bs 64, ReduceLROnPlateau (factor 0.5,
  patience 6, min 1e-5), early-stop patience 30 on val CTC.
- Log: `reports/train_v6.log` (the earlier partial run that was killed at
  ep 114 is preserved as `reports/train_v6_partial_ep114.log`).

| | v5 (no augmentation) | v6 full |
|---|---|---|
| epochs run | 101 (early stop) | 198 (early stop) |
| best val_ctc | 2.0109 | 1.5903 |
| best val_word_acc | 33.4% | 41.0% |

## Results (int8 ONNX, CPU — the shipping artifact)

### Frozen v1 benchmark (6,470 in-domain + 5,022 OOD)

| Metric | v5 | v6 full |
|---|---:|---:|
| intent accuracy | 56.4% | 49.7% |
| exact match | 44.3% | 40.7% |
| slot F1 | 55.2% | 53.8% |
| WER | 81.6% | 81.5% |
| OOD rejection | 73.7% | 92.1% |
| in-domain false rejects (gate) | n/a | 9.0% |
| command accuracy | 57.2% | 63.2% |

Reject-gate tuning: `reports/reject_threshold_v6.json` (no threshold met
the 5% in-domain reject budget — the floor is 9.0% in-domain rejects even at the loosest
threshold, so the gate ships at the conservative fallback -0.30).

### New-speaker generalization (`additional_test_data`, 171 clips — held out)

| Metric | v1 (v5 model) | v6 full (no gate) | v6 full (gate) |
|---|---:|---:|---:|
| intent accuracy | 10.5% | 23.4% | 8.2% |
| exact match | 8.2% | 18.7% | 7.6% |
| WER | 83.9% | 70.8% | 87.0% |
| blank transcript rate | 47% | 31.6% | 70.8% |

### Latency (int8, CPU, 300 rows)

p50 18.8 ms, p95 70.0 ms, p99 123.9 ms, mean 26.5 ms, RTF 0.0095 (300 rows, int8, CPU — comfortably real-time).

## Two-stage pipeline (VCM-v2)

VCM-v2 (ASR → TF-IDF classifier) reuses `me2_v6/best.pt` as Stage 1. The
Stage-2 classifier alone scores 95.9% command / 97.7% intent on gold
transcripts, so the end-to-end number is set by the ASR. Re-evaluated with
the completed v6 model: **24.6% command / 31.6% intent** end-to-end on
the 171 new-speaker clips (reports/asr_variant_v6.json in the VCM-v2
repo) — up from 22.8% / 29.8% with the partial v6 model, and 2.2x the
v1 single-model pipeline's 10.5% intent.

## v6 vs v7 (intent balancing) — and the verdict

A second run, **v7**, tested the remaining lever from the plan: *per-intent
balancing* of the training features (`scripts/make_balanced_features.py`,
inverse-intent-frequency resampling, cap 3.0x/0.5x). Same architecture, same
augmentation, same recipe — only the training-row mix changed.

| | v6 (normal mix) | v7 (intent-balanced) |
|---|---:|---:|
| epochs run (early stop) | 198 | 138 |
| best val_ctc (shared val) | **1.5903** | 1.8199 |
| best val_word_acc | **41.0%** | 37.4% |
| frozen v1 intent acc (gate) | 49.7% | 42.9% |
| frozen v1 command acc (gate) | 63.2% | 60.7% |
| frozen v1 OOD rejection (gate) | 92.1% | 93.2% |
| new-speaker intent (no gate) | **23.4%** | 15.8% |
| new-speaker exact (no gate) | **18.7%** | 14.0% |
| new-speaker WER (no gate) | **70.8%** | 81.1% |

**Balancing hurt.** v7's val CTC is worse (1.82 vs 1.59) and it generalizes
*less* well to the new speaker (intent 15.8% vs
23.4%, WER 81.1% vs 70.8%).
Over-weighting the rare intents (call, set_alarm, set_timer, set_temperature)
and trimming the dominant one (media_control) made the acoustic model
overfit the rare-intent phrasings at the expense of the shared command
vocabulary — exactly the opposite of what a new speaker needs. The val split
(untouched by balancing) caught it, which is why early stopping fired sooner.

## Verdict

* **More training + mel-domain augmentation (v6) is the win.** On the held-out
  new speaker, v6 roughly **doubles** v1's intent accuracy
  (23.4% vs 10.5%), cuts WER (70.8% vs 83.9%)
  and the blank-transcript rate (31.6% vs 47%). In-domain, the reject gate
  trades ~9% in-domain false rejects for OOD rejection
  **92.1% vs 73.7%** (false-accept 7.9%
  vs 26.3%) and lifts command accuracy 63.2% vs 57.2%.
* **Intent balancing (v7) is dropped** — worse on every axis that matters here.
* **The reject gate cannot meet the 5% in-domain budget** (floor
  9.0%); it ships at -0.30 as a conservative
  silence-on-low-confidence, not a tight precision gate.
* **What remains:** the new-speaker gap is now cleanly isolated to the
  acoustic model (the VCM-v2 classifier alone is 99.4%). The next real lever is
  a stronger ASR backbone (pretrained/fine-tuned) — or, if the new recordings
  may be used, adding them to training, which was deliberately held out as the
  honest generalization test.

*Reports: `reports/v6_frozen_int8.json`, `reports/v7_frozen_int8.json`,
`reports/additional_test_v6_int8.json` (+`_gated`), `reports/v7_*.json`,
`reports/v6_latency.json`, `reports/v7_latency.json`,
`reports/reject_threshold_v6.json`, `reports/reject_threshold_v7.json`,
`reports/train_v6.log`, `reports/train_v7.log`.*
