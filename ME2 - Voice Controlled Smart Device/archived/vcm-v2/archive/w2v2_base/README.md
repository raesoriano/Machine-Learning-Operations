# archive/w2v2_base/ — wav2vec2-base-960h CTC backbone (ABANDONED)

Frozen snapshot of the first attempt at the "pretrained ASR" backbone for
VCM-v2: **wav2vec2-base-960h fine-tuned with CTC over the 1033-word
constrained ME2 command vocabulary**. It does not work; it is kept here for
provenance. The active backbone is **Whisper base.en** (see `../../backbone/`).

## Why it was abandoned

CTC over a 1033-word vocabulary on ~2 s command clips is fundamentally
broken at this scale:

1. **Prior collapse.** The overfit probe
   (`scripts/overfit_probe.py`: train CTC on only 300 clips, which a working
   model must memorize) plateaus at val CTC loss ~135 and never gets below
   ~5. Every single clip decodes to the same one-word output, `"beatles"` —
   the model collapses to its prior instead of learning the vocabulary.
2. **Full runs confirm it.**
   - run 1 (lr 3e-5 constant, 15 epochs, best ep 10, val CTC 243.2):
     ME2 test WER **250.14%**, word exact match **0.0%**
     (`reports/train_report_run1_lr3e5.json`,
     `reports/asr_regression_w2v2_base.json`).
   - run 2 (lr 3e-4 + warmup + cosine, 6 epochs): val CTC rose 253.8 ->
     279.6 from epoch 1 — the model is destroyed, not learned
     (`artifacts/train_report.json`).
3. **Data and decode verified sound** (`scripts/diag_train.py`): 16 kHz
   wavs, ~2 s duration, 2–6 tokens per clip; the greedy CTC decoder
   (`model/decode.py` from the ME2 repo) is correct. The failure is the
   training objective/capacity, not the data.

End-to-end on the held-out 171-clip set it was never evaluated — the
backbone was abandoned after the training diagnostics above (zero-shot
Whisper base.en, at 81.9% command accuracy on the same set, made it
moot).

## Contents

| Path | Contents |
|---|---|
| `scripts/build_w2v2_data.py` | (kept in `../../backbone/scripts/`) built the JSONL data |
| `scripts/finetune_w2v2.py` | CTC fine-tune (run 1 config) |
| `scripts/finetune_w2v2_v2.py` | CTC fine-tune v2 (warmup/cosine; orphaned, never completed) |
| `scripts/eval_w2v2.py`, `scripts/eval_w2v2_split.py` | end-to-end evals |
| `scripts/overfit_probe.py` | the 300-clip memorization probe that exposed the collapse |
| `scripts/diag_train.py` | data/model diagnostics |
| `vcm2b/asr_w2v2.py` | ASR wrapper (greedy CTC decode + reject gate) |
| `artifacts/` | run-2 checkpoint (config.json, model.safetensors ~376 MB via git-lfs, vocab.json, train_report.json; best.pt/last.pt are intermediate) |
| `reports/` | run-1 train report + ME2 regression report |

## Reproduce (not recommended)

```bash
python backbone/scripts/build_w2v2_data.py --no-newspk
python archive/w2v2_base/scripts/finetune_w2v2.py --epochs 15 --batch-size 32
python archive/w2v2_base/scripts/eval_w2v2.py --data ../additional_test_data
```
