# pi test v7 — PocketSphinx ensemble (v3 architecture) with the custom AM **retrained on the v6 dataset**

v7 keeps the **v3 architecture** — the grammar-constrained PocketSphinx
ensemble (custom AM + stock `en-us` AM decoding the same 103-phrase JSGF
grammar, fused with the stage-2 TF-IDF/LR classifier) — but the **custom
acoustic model is retrained from scratch on the v6 dataset** (the same
dataset v6/v8 were built on). The retrain is a full `sphinxtrain` CD
(context-dependent) pipeline: LDA → MLLT → CI HMM → multipron → untied CD →
buildtrees → prunetree → **tied CD (16 densities)** → lattice.

So v7 answers: *does the classic PocketSphinx HMM stack, given the same data
as the neural v8, get closer?*

## Results — v6 test split (4418 clips, same protocol as v8)

| metric | v7 (this) | v8 (Conformer+CTC) | v6 HMM (real-only) |
|---|---:|---:|---:|
| overall acc | **0.7639** | 0.8497 | 0.429 |
| command acc (in-scope) | **0.7710** | 0.8588 | — |
| reject acc (OOS) | **0.1064** | 0.0 | — |
| intent acc | **0.7890** | 0.8717 | — |
| real overall | **0.3381** | 0.4019 | 0.138 |
| synthetic overall | **0.8967** | 0.9893 | 0.520 |
| latency p50 / p90 (ms) | **118.2 / 228.6** | 28.6 / 70.0 | — |

4418 clips = 4371 in-scope (19 coarse commands) + 47 out-of-scope
(REJECT). Real = 1050 clips, synthetic = 3368 clips.

### Per-class (v7)

| class | n | acc |
|---|---:|---:|
| ALARM | 423 | 0.9267 |
| BRIGHTNESS | 423 | 0.8700 |
| CALL | 141 | 0.6099 |
| COLOR | 423 | 0.5414 |
| CREATE_REMINDER | 423 | 0.8605 |
| LIGHT_OFF | 141 | 0.6809 |
| LIGHT_ON | 141 | 0.6241 |
| LIST_REMINDERS | 141 | 0.7447 |
| MESSAGE | 141 | 0.8723 |
| NEXT | 141 | 0.7660 |
| PAUSE | 141 | 0.7589 |
| PLAY_MUSIC | 141 | 0.5745 |
| REJECT | 47 | 0.1064 |
| STOP | 141 | 0.5532 |
| TEMPERATURE | 423 | 0.9196 |
| TIME | 141 | 0.7801 |
| TIMER | 423 | 0.9291 |
| VOLUME_DOWN | 141 | 0.5957 |
| VOLUME_UP | 141 | 0.5674 |
| WEATHER | 141 | 0.6312 |

## How it was built

1. **Retrain** — `sphinxtrain` (stages `000,01,02,10,11,12,20,21,22,30,40,45,50,60,61,62`)
   on the v6 dataset, `CFG_CD_TRAIN=yes`, `CFG_FINAL_NUM_DENSITIES=16`.
   Final model: `vcm.cd_cont_200` (tied CD, 16 densities, 4918 triphones,
   311 tied states).
2. **Package** — the 8 AM files (`feat.params`, `feature_transform`, `mdef`,
   `means`, `mixture_weights`, `noisedict`, `transition_matrices`,
   `variances`) are copied into `am/custom`, replacing v3's original AM. The
   v3 AM is preserved in `am/custom_v3`. The v3 grammar LM (`vcm.lm.bin` /
   `vcm.lm.arpa`) and `dict3` are kept unchanged.
3. **Eval** — `training/eval_v7.py --split test --workers 32` on the v6 test
   split (the same 4418 clips v8 was scored on), so v7 and v8 are directly
   comparable.

## What to note

* **v7 vs v8 (same data, same split):** v8's neural Conformer+CTC leads on
  overall (0.8497 vs 0.7639) and intent; v7's HMM ensemble is the
  classic, lighter, fully-interpretable baseline.
* **v7 vs v6 HMM (real-only):** v7's retrained CD AM is the v3 stack given
  the v6 data; v6's HMM baseline was 0.429 overall / 0.138 real.
* **v3's own 0.9375** was on a different, smaller 176-clip held-out protocol
  (one new speaker) — not comparable to the 4418-clip v6 test split used here.
* The retrain is a **CD** model (context-dependent), matching v3's deployed
  `am/custom` (which was itself a CD model), so the ensemble wiring is
  unchanged.

## Files

* `vcm_pi_v3.py` — the interactive loop (wake word + spoken responses),
  unchanged from v3.
* `am/custom/` — **v7 retrained CD AM** (16-dens) + v3 grammar LM.
* `am/custom_v3/` — the original v3 CD AM (backup).
* `am/stock/`, `am/stock_enus/` — stock `en-us` AM + cmudict (ensemble 2nd leg).
* `training/eval_v7.py` — the v6-test-split evaluator (same protocol as v8).
* `_eval_test.json` — full per-clip results for this run.
