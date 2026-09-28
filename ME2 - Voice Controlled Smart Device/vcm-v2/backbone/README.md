# backbone/ — pretrained Whisper ASR + 31-command classifier

The active VCM-v2 pipeline. Replaces the from-scratch 422k CTC encoder
(archived in `../archive/ctc_v8/`) with **pretrained Whisper base.en**
(openai/whisper-base.en, 72M, seq2seq) fine-tuned on the ME2 command
transcripts. (An intermediate wav2vec2-base-960h CTC attempt was abandoned
— broken objective — and is archived in `../archive/w2v2_base/`.)

## Why

The tiny CTC underfits even its own in-domain data (47.7% WER on the ME2
test split, 41.4% deletion rate). A pretrained seq2seq ASR already knows
English acoustics and general text; fine-tuning on the ME2 transcripts
teaches it the command vocabulary and the canonical text form the stage-2
classifier was trained on. Zero-shot (no fine-tune) it already scores
**81.9% command accuracy** on the 171-clip sole test set, vs 24.6% for
every from-scratch attempt; the v1 fine-tune lifts it to **85.4%** (best).
Stage 2 (the TF-IDF + LogReg 31-command classifier, 99.4% on gold
transcripts) is **unchanged and reused** from the archive, so all ASR
backends are directly comparable on the same 171-clip test set.

## Pipeline

```
audio (16 kHz)
  -> Whisper base.en (fine-tuned, seq2seq)
  -> normalize (digits <-> number-words)
  -> TF-IDF + LogReg classifier  (archive/ctc_v8/vcm2/classifier.py)
  -> one of 31 commands | REJECT
```

## Fine-tune v1 (BEST + active)

`scripts/finetune_whisper.py` — single LR 1e-5, CE-loss early-stop (patience
2, stopped epoch 3, best epoch 1). On the 171-clip sole test set it scores
**85.4% command / 86.0% intent** — the best ASR in the project (re-verified
2026-09-28, `reports/additional_test_whisper_ft_rerun.json`). Caveat: it
loops on 70/171 clips (40.9%) — mean WER 10.8%, median 0.0%; the stage-2
classifier still extracts the command from looped transcripts.

## Fine-tune v2 (staged schedule — tried, scored worse, archived)

`scripts/finetune_whisper_v2.py`. Hypothesis: v1's loops came from an
under-trained head, so v2 (1) **Stage A — warm-start, encoder FROZEN**
(3 epochs, decoder-only, LR 3e-5), (2) **Stage B — encoder UNFROZEN**
(up to 12 epochs, encoder LR 1e-5 / decoder LR 3e-5, cosine, 6% warmup),
(3) **early stop on validation WER** (300 sampled clips, patience 3),
(4) bf16. Result: best checkpoint = Stage A epoch 1 (val WER 6.20),
early-stopped after 4 epochs. On the 171-clip test set it scored **50.3%
command / 55.0% intent with 66.7% looping** — worse than v1 and worse than
zero-shot. The staged recipe did not help; the val-WER early-stop picked a
checkpoint that had barely moved from the pretrained weights. Kept for
reference; **v1 remains the active backbone**.

## Layout

| Path | Contents |
|---|---|
| `vcm2b/asr_whisper.py` | ASR wrapper: faster-whisper base.en (zero-shot path; `condition_on_previous_text=False`, no-speech 0.6, 200-word cap) |
| `scripts/build_w2v2_data.py` | builds `data/{train,val}.jsonl` (raw 16 kHz wav + transcript) from the ME2 manifest; `--no-newspk` = leakage-free |
| `scripts/finetune_whisper.py` | **active** v1 seq2seq fine-tune (single LR, CE early-stop) — best model |
| `scripts/finetune_whisper_v2.py` | staged fine-tune (frozen-encoder warm-start → unfrozen cosine; val-WER early-stop) — tried, scored worse, kept for reference |
| `scripts/eval_whisper_robust.py` | zero-shot end-to-end eval on `additional_test_data` (persistent worker + per-clip timeout) |
| `scripts/eval_whisper_ft.py` | fine-tuned end-to-end eval on `additional_test_data` (HF `model.generate`) |
| `scripts/eval_whisper_ft_me2.py` | in-domain regression guardrail: ME2 TEST split (original speakers, never fine-tuned on), WER + exact match |
| `artifacts/whisper_base_ft/` | **BEST + active** fine-tuned HF model (`best/`, `last/`, train_report.json) — weights via git-lfs |
| `artifacts/whisper_base_ft_v2/` | staged v2 fine-tune (scored worse; kept for reference) — weights via git-lfs |
| `archive/whisper_ft_v1/` | **archived copy** of the v1 fine-tune (BEST; see its README) — model + reports |
| `reports/` | eval result JSONs |

## Data (leakage-free)

- **train**: ME2 optionb positives, train split (37,992 rows)
- **val**: ME2 optionb positives, val split (5,315 rows)
- **test (held out, never trained)**: the 171 RAW `additional_test_data`
  clips — one new speaker. **Every derived variant** (denoised, noise,
  reverb, pitch, stretch, hp80) is excluded from train and val as well,
  so the held-out set is genuinely unseen. Built with
  `build_w2v2_data.py --no-newspk` (the name is historical).

## Reproduce

```bash
python backbone/scripts/build_w2v2_data.py --no-newspk
# v1 fine-tune (BEST + active):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/finetune_whisper.py \
    --epochs 6 --batch-size 32 --lr 1e-5 \
    --out backbone/artifacts/whisper_base_ft
# v2 staged fine-tune (tried, scored worse):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/finetune_whisper_v2.py \
    --out backbone/artifacts/whisper_base_ft_v2
# eval (sole test set — local copy, git-ignored):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft.py \
    --model backbone/artifacts/whisper_base_ft/best \
    --data test_data/additional_test_data \
    --report backbone/reports/additional_test_whisper_ft.json
# in-domain regression guardrail (ME2 test split):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft_me2.py \
    --model backbone/artifacts/whisper_base_ft/best \
    --n 400 --seed 42 \
    --report backbone/reports/me2_regression_whisper_ft.json
# zero-shot baseline (faster-whisper):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_robust.py \
    --data test_data/additional_test_data \
    --report backbone/reports/additional_test_whisper.json
```

## Results (sole test set: 171 clips, `test_data/additional_test_data`)

| Stage 1 | Command (171 held-out) | Intent | WER vs spoken |
|---|---|---|---|
| whisper base.en zero-shot | 81.9% | 81.9% | 22.5% |
| whisper base.en fine-tuned **v1** (BEST + active) | **85.4%** | **86.0%** | 10.8% mean / 0% median (40.9% looping) |
| whisper base.en fine-tuned **v2** (staged, archived) | 50.3% | 55.0% | 18.6% mean / 19% median (66.7% looping) |

In-domain regression guardrail (ME2 test split, 400 clips, original
speakers — never fine-tuned on):

| Model | WER | word-exact-match |
|---|---|---|
| v1 fine-tune (BEST + active) | 5.25 | 6% |
| v2 fine-tune (archived) | not run | — |

See the top-level README for the full comparison table.
