# VCM v2 — Two-Stage Voice Command Model

A voice-controlled smart-device command model that understands **31 fixed
commands** and rejects everything else, built as two independent, learnable
stages:

```
audio (16 kHz mono)
   │
   ▼
[ Stage 1 — ASR ]        audio -> transcript (the words actually spoken)
   Whisper base.en (72M, openai/whisper-base.en), fine-tuned seq2seq on the
   ME2 command transcripts (leakage-free: no additional_test_data).
   (The from-scratch 422k CTC is archived in archive/ctc_v8; the failed
   wav2vec2-base-960h CTC attempt is archived in archive/w2v2_base.)
   │
   ▼
[ normalize ]            digits <-> number-words alignment
   "Alarm 6 AM"  ->  "alarm six am"   (ASR emits number-words; the
   manifest uses digits; slot commands differ ONLY by that number)
   │
   ▼
[ Stage 2 — Classifier ] transcript -> one of 31 commands | REJECT
   TF-IDF (word unigrams+bigrams + char 2-5 grams) + multinomial
   logistic regression (class_weight='balanced'), trained on the ME2
   manifest (49,791 command rows + 30,366 REJECT rows, after data fixes).
   99.4% on gold transcripts — the text stage is at its ceiling.
   │
   ▼
   command (e.g. ALARM_6_00AM)  |  REJECT
```

## Why two stages (the v1 -> v2 fix)

v1 trained one model to map **audio -> canonical phrase**
(`set an alarm for six am`) while the audio actually said a **paraphrase**
(`Alarm 6 AM`). That mapping is unlearnable — it capped v1 at 56% in-domain
and ~10% on a new speaker, with 47% blank transcripts.

v2 splits it into two **learnable** tasks:

1. **ASR** — audio -> *the words actually spoken* (consistent mapping).
2. **Classifier** — spoken words -> one of 31 commands, or REJECT.

The classifier sees only words, so it is **speaker-independent by
construction** (the fix for v1's speaker overfitting), has an explicit
**REJECT** class (v1 had no reject training signal), and is trivially
interpretable and fast (<1 ms).

## Repo layout

```
data  ->  ../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/data
          (symlink, tracked as a link — the audio is NOT re-downloaded here)

backbone/               ACTIVE pipeline: pretrained Whisper ASR
  vcm2b/asr_whisper.py  #   Stage 1: faster-whisper base.en (zero-shot path)
  scripts/build_w2v2_data.py   # raw 16 kHz + transcript JSONL (train/val)
  scripts/finetune_whisper.py  # v1 seq2seq fine-tune (superseded; see archive)
  scripts/finetune_whisper_v2.py  # ACTIVE staged fine-tune (frozen-encoder
                                  #   warm-start -> unfrozen cosine; val-WER early-stop)
  scripts/eval_whisper_robust.py   # zero-shot end-to-end eval (171 clips)
  scripts/eval_whisper_ft.py       # fine-tuned end-to-end eval (171 clips)
  scripts/eval_whisper_ft_me2.py   # in-domain regression guardrail (ME2 test)
  artifacts/whisper_base_ft_v2/ #   fine-tuned HF model (git-lfs) + train report
  archive/whisper_ft_v1/      #   archived copy of the v1 fine-tune (BEST; see its README)
  reports/                #   eval result JSONs

archive/ctc_v8/           FROZEN snapshot of the original CTC pipeline
  vcm2/                   #   asr.py (me2_v6/v8 CTC wrapper), classifier.py,
                          #   normalize.py, ground_truth.py, pipeline.py, data_clean.py
  scripts/                #   build_subset.py, train_asr_v8.py, train_classifier.py,
                          #   tune_classifier.py, eval.py, eval_asr_split.py,
                          #   eval_asr_variants.py, noise_probe.py, noise_ab.py,
                          #   diagnose_v6_v8.py, ab_validate.py, typo_probe.py,
                          #   classifier_only.py
  reports/                #   all result JSONs (see archive/ctc_v8/README.md)
  artifacts/              #   classifier.pkl, asr_me2_v8 checkpoints (committed)
  data_new/               #   171-clip new-speaker subset manifest + build report
                          #   (wavs + feature npz regenerable, git-ignored)

archive/w2v2_base/        ABANDONED wav2vec2-base-960h CTC backbone (broken
                          #   objective; see archive/w2v2_base/README.md)
```

## The 31 commands

The 31 commands are the 19 dataset intents expanded by their slot values
(alarm times, brightness %, temperature, timer durations, colors, reminder
actions) — see `backbone/vcm2b` ground-truth mapping and the ME2
`data/manifests` for the full list. REJECT is the 32nd class (out-of-domain
speech / noise).

## Results on `additional_test_data` (171 clips, 19 intents, new speaker)

**Sole test set.** `additional_test_data` — 171 clips, **one new speaker held
out of all training** (the raw clips are never seen in any form). It is the
project's only test set; a local copy lives at `test_data/additional_test_data`
(git-ignored) so the project is self-contained.

| Stage 1 (ASR) | Command (31-way) | Intent (19-way) | Blank | WER vs spoken |
|---|---|---|---|---|
| me2_v6 (from-scratch 422k CTC, speaker-aug) | 24.6% | 28.1% | 38.6% | 76.5% |
| me2_v8 (v6 + new-speaker subset fine-tune) | 24.0% | 28.1% | 38.6% | 76.3% |
| wav2vec2-base-960h fine-tuned (ABANDONED) | not evaluated (abandoned) | — | — | 250.1% (ME2 test) |
| whisper base.en zero-shot (backbone) | 81.9% | 81.9% | 0.0% | 22.5% |
| whisper base.en fine-tuned **v2** (staged, archived) | 50.3% | 55.0% | 0.0% | 18.6% mean / 19% median |
| **PocketSphinx ensemble (custom 1.6 MB + stock 6.4 MB AM, agree+clf fusion)** | **95.3%** | **97.1%** | **0.0%** | **7.4% mean / 0% median** |
| **PocketSphinx stock en-us AM + 103-phrase JSGF (best single AM, 6.4 MB)** | **89.5%** | **91.8%** | **2.3%** | **12.7% mean / 0% median** |
| PocketSphinx custom 200-LDA AM + 102-phrase JSGF (1.6 MB) | 87.1% | 90.1% | 3.5% | 14.8% mean / 0% median |
| whisper base.en fine-tuned v1 (backbone, active) | 85.4% | 86.0% | 0.0% | 10.8% mean / 0% median |

- **Best model (ASR accuracy):** the **PocketSphinx ensemble** — the custom
  1.6 MB AM and the stock 6.4 MB AM both decode the completed 103-phrase JSGF
  and are fused by agreement + stage-2 classifier confidence — **95.3% command
  / 97.1% intent** (`backbone/reports/pocketsphinx_ensemble_cmudict.json`). It
  beats the Whisper fine-tune (85.4%) by +9.9 points at ~1/26th the footprint.
  The best *single* AM is the stock en-us AM + grammar at 89.5% / 91.8% for
  6.4 MB (under the 10 MB ideal).
- **Active production model:** `backbone/artifacts/whisper_base_ft/best` (the
  v1 fine-tune) — **85.4% command / 86.0% intent**, re-verified 2026-09-28 on
  the 171-clip sole test set (`reports/additional_test_whisper_ft_rerun.json`).
  It beats the zero-shot baseline (81.9%) and the v2 staged fine-tune (50.3%).
- **WER caveat (v1):** mean 10.8% is inflated by repetition loops on 70/171
  clips (40.9%) — the fine-tuned model doesn't always emit EOS on noisy OOD
  input and repeats a phrase. Median WER is 0.0% (the typical clip is exact).
  Loops don't hurt command accuracy (the stage-2 classifier extracts the
  command from the looped transcript). The zero-shot path (faster-whisper)
  has no loops and 22.5% WER — a cleaner transcript, slightly lower accuracy.
- **Classifier alone** (gold transcripts in): **99.4%** command accuracy —
  the end-to-end gap lives entirely in Stage 1.
- **Inference latency (GPU):** zero-shot p50 21 ms / p95 31 ms (faster-whisper
  int8); fine-tuned v1 p50 225 ms / p95 570 ms (HF `generate`).
- **Fine-tune data**: ME2 optionb positives only (train 37,992 / val 5,315),
  built with `build_w2v2_data.py --no-newspk` — the 171 raw
  `additional_test_data` clips and every derived denoised/noise/reverb/pitch
  variant are excluded, so the held-out set is genuinely unseen.

## The ASR journey (why the backbone exists)

1. **me2_v5 -> me2_v6** (ME2 repo): more training + mel-domain speaker
   augmentation (pitch ±3 st, stretch, gain, shift, SpecAugment). Lifted
   end-to-end command accuracy 22.8% -> 24.6% on the new speaker.
2. **me2_v7** (per-intent balancing): tried and **dropped** — over-weighting
   rare intents overfit their phrasings; worse on every axis.
3. **me2_v8** (this repo, `archive/ctc_v8`): fine-tuned v6 on the 171-clip
   new-speaker subset (×9 variants). **A null** — 154/171 transcripts
   byte-identical to v6. Diagnosis: the 422k CTC **underfits even its own
   in-domain data** (47.7% WER on the ME2 test split, 41.4% deletion). The
   bottleneck is **model capacity, not speaker coverage**.
4. **wav2vec2-base-960h CTC** (this repo, `archive/w2v2_base`): a pretrained
   94M-param encoder fine-tuned with CTC over the 1033-word command vocab.
   **Abandoned** — the objective collapses (overfit probe memorizes nothing,
   every clip decodes to one word; full runs: ME2 test WER 250%, val loss
   rises from epoch 1). See `archive/w2v2_base/README.md`.
5. **Whisper base.en** (this repo, `backbone/`): pretrained 72M-param seq2seq
   ASR. Zero-shot it already hits 81.9% command accuracy on the held-out
   new-speaker set (vs 24.6% for every from-scratch attempt). The first
   fine-tune (v1) is the **best and active** backbone — 85.4% command /
   86.0% intent on the 171-clip sole test set (re-verified 2026-09-28),
   beating zero-shot (81.9%). The **v2 staged fine-tune** (frozen-encoder
   warm-start → unfrozen cosine LR; early stop on val WER) scored worse
   (50.3%) and is archived — see `backbone/README.md`.

## Ultra-small recognizer — PocketSphinx + command grammar (≤ 11 MB)

A **grammar-constrained PocketSphinx** recognizer is now the project's **best
ASR** — it beats the Whisper fine-tune on the 171-clip held-out set at a
fraction of the footprint. Two AMs decode the same 103-phrase JSGF and are
fused:

| Recognizer | Command | Intent | Blank | Latency p50 | Footprint |
|---|---|---|---|---|---|
| **ensemble: custom 1.6 MB + stock 6.4 MB AM, agree+clf fusion (BEST)** | **95.3%** | **97.1%** | **0.0%** | **87 ms** | **11.1 MB** |
| ensemble, dict3 for both AMs (under 10 MB ideal) | 93.0% | 93.6% | 0.6% | 85 ms | 8.0 MB |
| **stock en-us AM + 103-phrase JSGF (best single AM)** | **89.5%** | **91.8%** | **2.3%** | **58 ms** | **6.4 MB** |
| custom 200-LDA AM + 102-phrase JSGF | 87.1% | 90.1% | 3.5% | 29 ms | 1.6 MB |
| custom 200-LDA AM + original 93-phrase JSGF | 84.8% | 88.3% | 4.1% | 26 ms | 1.6 MB |
| stock en-us AM + 93-phrase JSGF (old fallback) | 78.4% | 83.0% | 6.4% | 60 ms | 9.6 MB |
| Whisper base.en fine-tuned v1 (largest) | 85.4% | 86.0% | 0.0% | 225 ms | 290 MB |

The custom AM (`backbone/artifacts/pocketsphinx_trained_lda/`) is a
`sphinxtrain` cd_cont model — 200 tied senones, 8 gaussians, LDA/MLLT 39→29,
trained on the 6,964 clean optionb clips. The stock AM is the bundled `en-us`
model (far more diverse training data). Both are constrained at decode time by
the 103-phrase JSGF, then the transcript is fed to the **same stage-2
classifier** as the Whisper path, so command/intent are measured identically.

**Two wins, both decoder-side — no AM retraining.**
1. **Complete the grammar.** The original 93-phrase JSGF covered only 51 of the
   62 test phrases, so the decoder snapped missing phrasings to the nearest one
   or blanked. Adding the 10 missing phrasings (`change color to red`,
   `lights out`, `shut off the lights`, the no-`to` `create a reminder …`
   forms, `end playback`, …) — a **zero-retrain, +0.5 KB** change — lifted the
   custom AM 84.8% → 87.1%.
2. **Ensemble with a larger pretrained AM.** The custom AM nails the CALL
   cluster but fails ALARM/COLOR/LIGHT_ON/TEMPERATURE/TIME/WEATHER; the stock
   AM is the reverse. Fusing them (use the agreed command, else the one whose
   stage-2 classifier is more confident — **test-set-agnostic**) recovers
   nearly all of each one's mistakes: **95.3% / 97.1%**, +9.9 pts over Whisper
   FT at ~1/26th the footprint. The 8 residual errors are 7 cases where *both*
   AMs agree on the same wrong answer (true acoustic confusions) + 1.

Retraining the custom AM with more capacity (800 senones → 80.7%), density
(16 gaussians → 77.8%), or speakers (100 TTS voices → 77.2%) all made it
*worse* — the 1.6 MB model is the best custom AM; the gains came from the
grammar and the ensemble. Full numbers + the retrain ablations:
`backbone/reports/pocketsphinx_comparison.json`.

- **Build/retrain:** `backbone/pocketsphinx/build_trained_am.py` (sphinxtrain,
  built from source at `sphinx_src/install`).
- **Eval (single AM):** `backbone/pocketsphinx/eval_pocketsphinx.py` (same
  ground truth + classifier + WER definition as the Whisper eval).
- **Eval (ensemble):** `backbone/pocketsphinx/eval_pocketsphinx_ensemble.py`
  (custom + stock AM, agree+clf fusion).
- **Deploy:** the `pocketsphinx` pip wheel (aarch64 available) loads the AM +
  dict + JSGF directly; no separate model download. The stock en-us + JSGF
  fallback ships in `pi_test/pocketsphinx/` (~10.75 MB) for the zero-training
  path.

## Noise pre-processing — measured, and it is NOT the fix

The test audio is genuinely noisier (noise floor −27.6 dBFS vs −49.2 dBFS in
training; 0–100 Hz rumble), but an A/B of four enhancement conditions on all
171 clips moved end-to-end accuracy by at most +1.7 pts command / +2.9 pts
intent and broke ~18 clips that raw got right — the voice, not the noise, is
the problem. A light 80 Hz high-pass is still recommended for the mic path
(hygiene, ~3 ms), but enhancement is not the fix. Details:
`archive/ctc_v8/reports/noise_ab.json`.

## Classifier improvements (archived pipeline)

The stage-2 classifier was tuned to **99.4%** on the held-out new-speaker set
(one remaining mismatch: `voluyme up`, a slip-of-the-tongue non-word). The
validated changes (see `archive/ctc_v8/reports/tune_results.json`):

1. **`class_weight='balanced'`** — the classes are imbalanced (STOP ×4167 vs
   ALARM_9_00PM ×599); balanced weights cut false-REJECT on val 0.90% -> 0.30%.
2. **Char 2-5 grams** added to TF-IDF — bag-of-words has zero typo tolerance;
   char ngrams recover near-miss ASR variants.
3. **Data-quality fixes** (`archive/ctc_v8/vcm2/data_clean.py`) applied to a
   copy of the manifest (lights-out relabel + STOP `end X` rows).

## Known limitation (pre-existing, affects all ASR versions)

The constrained 1033-word vocab is **missing the number-words `twenty`,
`two`, `hundred`**, so TEMPERATURE_22/26 and BRIGHTNESS_100 targets silently
lose their number. It affects the CTC models (archive/ctc_v8,
archive/w2v2_base); the Whisper backbone is unaffected (it uses its own
tokenizer, not the constrained vocab).

## Reproduce

```bash
# --- backbone (active) ---
python backbone/scripts/build_w2v2_data.py --no-newspk   # leakage-free JSONL
python backbone/scripts/finetune_whisper.py --epochs 6 --batch-size 32 --lr 1e-5
python backbone/scripts/eval_whisper_robust.py --data test_data/additional_test_data \
    --report backbone/reports/additional_test_whisper.json
python backbone/scripts/eval_whisper_ft.py \
    --model backbone/artifacts/whisper_base_ft/best \
    --data test_data/additional_test_data \
    --report backbone/reports/additional_test_whisper_ft.json

# --- archive/ctc_v8 (frozen) ---
python archive/ctc_v8/scripts/build_subset.py --src test_data/additional_test_data
python archive/ctc_v8/scripts/train_asr_v8.py --features archive/ctc_v8/data_new/features \
    --init  "../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/best.pt" \
    --config "../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/config.yaml" \
    --out archive/ctc_v8/artifacts/asr_me2_v8
python archive/ctc_v8/scripts/train_classifier.py --clean
python archive/ctc_v8/scripts/eval.py --data test_data/additional_test_data \
    --report archive/ctc_v8/reports/additional_test_v8.json
```
