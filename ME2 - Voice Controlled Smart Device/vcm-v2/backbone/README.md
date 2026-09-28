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
**81.9% command accuracy** on the held-out 171-clip new-speaker set, vs
24.6% for every from-scratch attempt. Stage 2 (the TF-IDF + LogReg
31-command classifier, 99.4% on gold transcripts) is **unchanged and
reused** from the archive, so all ASR backends are directly comparable on
the same held-out 171-clip test set.

## Pipeline

```
audio (16 kHz)
  -> Whisper base.en (fine-tuned, seq2seq)
  -> normalize (digits <-> number-words)
  -> TF-IDF + LogReg classifier  (archive/ctc_v8/vcm2/classifier.py)
  -> one of 31 commands | REJECT
```

## Fine-tune v2 (active) — staged schedule, WER early-stop

`scripts/finetune_whisper_v2.py`. The v1 fine-tune
(`archive/whisper_ft_v1/`) was **degenerate**: single LR 1e-5, early stop on
**CE loss** (patience 2, stopped epoch 3). CE is a bad early-stopping
signal for seq2seq ASR — the model drove CE down by **hallucinating fluent
filler after the first word** (1024% WER on the held-out set; in-domain
guardrail WER 5.25, 6% exact match). v2 fixes the training recipe:

1. **Stage A — warm-start, encoder FROZEN** (3 epochs, decoder-only,
   LR 3e-5): teaches the head the domain text forms without disturbing the
   pretrained encoder.
2. **Stage B — encoder UNFROZEN** (up to 12 epochs, encoder LR 1e-5 /
   decoder LR 3e-5, cosine schedule, 6% linear warmup): adapts the encoder
   to the domain at a much lower LR.
3. **Early stop on validation WER** (300 sampled clips, greedy generation,
   patience 3) — best checkpoint = lowest val WER, not lowest CE.
4. bf16 mixed precision (A100).

Data is identical to v1 (leakage-free, `--no-newspk`).

## Layout

| Path | Contents |
|---|---|
| `vcm2b/asr_whisper.py` | ASR wrapper: faster-whisper base.en (zero-shot path; `condition_on_previous_text=False`, no-speech 0.6, 200-word cap) |
| `scripts/build_w2v2_data.py` | builds `data/{train,val}.jsonl` (raw 16 kHz wav + transcript) from the ME2 manifest; `--no-newspk` = leakage-free |
| `scripts/finetune_whisper.py` | v1 seq2seq fine-tune (single LR, CE early-stop) — kept for reference; **superseded by v2** |
| `scripts/finetune_whisper_v2.py` | **active** staged fine-tune (frozen-encoder warm-start → unfrozen cosine; val-WER early-stop; bf16) |
| `scripts/eval_whisper_robust.py` | zero-shot end-to-end eval on `additional_test_data` (persistent worker + per-clip timeout) |
| `scripts/eval_whisper_ft.py` | fine-tuned end-to-end eval on `additional_test_data` (HF `model.generate`) |
| `scripts/eval_whisper_ft_me2.py` | in-domain regression guardrail: ME2 TEST split (original speakers, never fine-tuned on), WER + exact match |
| `artifacts/whisper_base_ft_v2/` | **active** fine-tuned HF model (`best/`, `last/`, feature extractor, tokenizer, train_report.json) — weights via git-lfs |
| `archive/whisper_ft_v1/` | **archived** v1 fine-tune (degenerate; see its README) — model + reports |
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
# v2 staged fine-tune (active):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/finetune_whisper_v2.py \
    --out backbone/artifacts/whisper_base_ft_v2
# eval:
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft.py \
    --model backbone/artifacts/whisper_base_ft_v2/best \
    --data ../additional_test_data \
    --report backbone/reports/additional_test_whisper_ft_v2.json
# in-domain regression guardrail (ME2 test split):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft_me2.py \
    --model backbone/artifacts/whisper_base_ft_v2/best \
    --n 400 --seed 42 \
    --report backbone/reports/me2_regression_whisper_ft_v2.json
# zero-shot baseline (faster-whisper):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_robust.py \
    --data ../additional_test_data \
    --report backbone/reports/additional_test_whisper.json
```

## Results

| Stage 1 | Command (171 held-out) | Intent | WER vs spoken |
|---|---|---|---|
| whisper base.en zero-shot | 81.9% | 81.9% | 22.5% |
| whisper base.en fine-tuned **v1** (archived) | 75.4% | 77.8% | 1024% (degenerate) |
| whisper base.en fine-tuned **v2** (active) | __CMD__% | __INT__% | __WER__% |

In-domain regression guardrail (ME2 test split, 400 clips, original
speakers — never fine-tuned on):

| Model | WER | word-exact-match |
|---|---|---|
| v1 fine-tune (archived) | 5.25 | 6% |
| v2 fine-tune (active) | __GWER__ | __GEM__% |

See the top-level README for the full comparison table.
