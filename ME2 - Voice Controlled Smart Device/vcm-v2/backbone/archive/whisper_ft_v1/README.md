# whisper_ft_v1 — archived (degenerate fine-tune)

Frozen snapshot of the **first** Whisper base.en fine-tune
(`backbone/scripts/finetune_whisper.py`), kept for reference. It is
**superseded by the v2 staged fine-tune** (see `backbone/README.md` and
`backbone/scripts/finetune_whisper_v2.py`).

## What this was

- Base: `openai/whisper-base.en` (72.6M), seq2seq teacher-forcing on the
  ME2 optionb transcripts.
- Schedule: single LR 1e-5 (AdamW), 6 epochs max, batch 32, early stop on
  **CE loss** (patience 2) → stopped at epoch 3 (val CE 0.131).
- Data: leakage-free (`--no-newspk`); the 171 `additional_test_data`
  clips were never in train/val.

## Why it was archived (the failure)

CE loss is a bad early-stopping signal for seq2seq ASR. The model drove
CE down by **hallucinating fluent filler after the first few words**
while still being useless:

| Metric (held-out 171 new-speaker clips) | v1 fine-tune | zero-shot baseline |
|---|---|---|
| command acc (31-way) | 75.4% | **81.9%** |
| intent acc (19-way) | 77.8% | **81.9%** |
| mean WER | **10.24 (1024%)** | 0.225 (22.5%) |
| blank rate | 0.0% | 0.0% |

The 75.4% command accuracy survives only because the stage-2 TF-IDF
classifier is robust to the hallucinated filler — the model gets the
first word(s) right and the classifier ignores the rest. In-domain
guardrail (ME2 test, 400 clips): WER 5.25, word-exact-match 6%.

Example failures (spoken → hypothesis):
- `call` → `call me now for a new video`
- `lights out` → `lights out now i am not going to play that one too much for me to play this week ...` (31 words)
- `next song` → `next song please play my best song for me please play me my best songs ... playlist playlist playlist ...` (28 words)

## Layout

- `model/` — the HF model dirs (`best/`, `last/`) + `train_report.json`
  (git-lfs for the .safetensors).
- `reports/additional_test_whisper_ft.json` — 171-clip eval (per-clip).
- `reports/me2_regression_whisper_ft.json` — in-domain guardrail.

Reproduce: `python backbone/scripts/finetune_whisper.py` (the v1 script
is still in `backbone/scripts/`).
