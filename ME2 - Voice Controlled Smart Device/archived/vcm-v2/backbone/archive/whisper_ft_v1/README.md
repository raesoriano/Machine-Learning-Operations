# whisper_ft_v1 — the BEST + active fine-tune (archived copy)

Frozen copy of the **best** Whisper base.en fine-tune
(`backbone/scripts/finetune_whisper.py`), kept alongside the active
`backbone/artifacts/whisper_base_ft/` (the two `best/` weight files are
byte-identical — same md5). It is **not** superseded: the v2 staged
fine-tune scored worse (50.3% vs 85.4%), so v1 remains the active
backbone. See `backbone/README.md`.

## What this is

- Base: `openai/whisper-base.en` (72.6M), seq2seq teacher-forcing on the
  ME2 optionb transcripts.
- Schedule: single LR 1e-5 (AdamW), 6 epochs max, batch 32, early stop on
  **CE loss** (patience 2) → stopped at epoch 3 (best epoch 1, val CE 0.128).
- Data: leakage-free (`--no-newspk`); the 171 `additional_test_data`
  clips were never in train/val.

## Results (171-clip sole test set, re-verified 2026-09-28)

| Metric (171 held-out new-speaker clips) | v1 fine-tune | zero-shot baseline |
|---|---|---|
| command acc (31-way) | **85.4%** | 81.9% |
| intent acc (19-way) | **86.0%** | 81.9% |
| mean WER | 10.8% (median 0.0%) | 22.5% |
| looping clips | 70/171 (40.9%) | 0 |
| blank rate | 0.0% | 0.0% |
| ASR latency p50 / p95 | 225 / 570 ms | 21 / 31 ms |

In-domain guardrail (ME2 test split, 400 clips, original speakers):
WER 5.25, word-exact-match 6%.

## Known caveat: repetition loops

The model does not always emit EOS on noisy OOD input and repeats a
phrase (40.9% of clips loop; mean WER 10.8%, median 0.0%). The stage-2
TF-IDF classifier is robust to the loops — it extracts the command from
the looped transcript, which is why command accuracy stays at 85.4%.
Example: `call` → `call me now for a new video` (still classified CALL).
The zero-shot path (faster-whisper) has no loops and 22.5% WER — a
cleaner transcript at a slightly lower accuracy.

> Historical note: an earlier eval of these same weights reported 75.4%
> command / 77.8% intent (12:15 run, `reports/additional_test_whisper_ft.json`
> below). The 14:05 and 2026-09-28 re-runs both report 85.4% / 86.0% —
> the current, verified numbers. The "degenerate / 1024% WER" label in
> older READMEs referred to this early run and has been retracted.

## Layout

- `model/` — the HF model dirs (`best/`, `last/`) + `train_report.json`
  (git-lfs for the .safetensors).
- `reports/additional_test_whisper_ft.json` — 171-clip eval, early run
  (75.4%; per-clip).
- `reports/me2_regression_whisper_ft.json` — in-domain guardrail.

Reproduce: `python backbone/scripts/finetune_whisper.py` (the v1 script
is still in `backbone/scripts/`).
