# vcm-v2 — migrated from `raesoriano/VCM-v2`

**2026-09-28:** the standalone `VCM-v2` repo is retired. All of its progress
(code, models, reports) is migrated here into the ME2 subfolder so the whole
voice-command project lives in one repo. The `VCM-v2` GitHub repo is kept as a
read-only snapshot (last commit `9d99a35`) and will no longer receive updates.

## What this is

The **VCM v2** two-stage voice command model: a **pretrained Whisper base.en**
ASR (stage 1) + a **TF-IDF + LogReg 31-command classifier** (stage 2, reused
from the CTC pipeline). It supersedes the from-scratch 422k CTC encoder in
`../model` + `../vcm` (me2_v5/v6/v7), which underfits even its own in-domain
data (47.7% WER on the ME2 test split).

## Layout

| Path | Contents |
|---|---|
| `README.md` | full pipeline docs, the ASR journey, results tables |
| `backbone/` | **active** Whisper pipeline: `vcm2b/asr_whisper.py` (faster-whisper zero-shot), `scripts/` (build data, fine-tune v1/v2, evals), `artifacts/` (fine-tuned models, git-lfs), `reports/` (eval JSONs), `archive/whisper_ft_v1/` (archived v1 fine-tune) |
| `archive/ctc_v8/` | frozen snapshot of the from-scratch CTC pipeline (vcm2 code, me2_v8 checkpoints, classifier.pkl, all reports) |
| `archive/w2v2_base/` | frozen snapshot of the abandoned wav2vec2-base-960h CTC attempt |
| `data` | **symlink → `../data`** (the ME2 dataset, tracked by this repo) |

## Data & leakage

- `data/` is the ME2 dataset (this repo's `data/` dir) — the same source the
  CTC pipeline used.
- The **171 raw `additional_test_data` clips** (one new speaker, recorded
  2026-09-27) are the held-out test set. They live in
  `../data/additional_test_data/` (ME2 `data/` dir, **on disk, untracked** —
  ME2's `data/.gitignore` keeps all audio on shared storage, never in git).
  They were **never** in any train/val set — every derived variant (denoised,
  noise, reverb, pitch, stretch) is excluded too.
- Eval scripts take the held-out set via `--data ../data/additional_test_data`
  (relative to `backbone/`). The `data` symlink in this folder points at
  `../data` (the ME2 dataset).

## Results (171 held-out new-speaker clips, stage-1 + stage-2 end-to-end)

| Stage 1 | Command | Intent | WER (median / mean) |
|---|---|---|---|
| whisper base.en **zero-shot** | 81.9% | 81.9% | 22.5% / 22.5% |
| whisper base.en **fine-tuned v1** (active) | **85.4%** | **86.0%** | 0% / 1084%* |
| whisper base.en fine-tuned v2 (staged) | 50.3% | 54.9% | 19% / 1861% |

\* v1 repetition-loops on 40.9% of clips (no EOS on OOD/noisy input), which
inflates the *mean* WER; the median is 0 and command accuracy is the highest
of all attempts. v2's staged recipe (frozen-encoder warm-start → unfrozen
cosine, val-WER early-stop) did **not** fix the degeneration — it loops more
(66.7%) and scores worse on every axis. **v1 is the active model.**

## Git notes

- Model weights are **git-lfs** tracked (`.gitattributes` in this folder):
  `backbone/**/model.safetensors` (4 unique objects, ~277 MB each) and
  `archive/w2v2_base/artifacts/model.safetensors` (364 MB). The v1 model
  appears in two places (`backbone/artifacts/whisper_base_ft/` and
  `backbone/archive/whisper_ft_v1/model/`) but is the **same LFS object**
  (deduplicated).
- `archive/ctc_v8/artifacts/asr_me2_v8/*.pt` (3.4 MB) are force-added: the
  repo-root `.gitignore` ignores `*.pt` globally.
- Regenerable artifacts are NOT committed (same policy as VCM-v2):
  `archive/ctc_v8/data_new/features/` (1.2 GB npz — rebuild with
  `archive/ctc_v8/scripts/build_subset.py`), `archive/w2v2_base/artifacts/*.pt`
  (raw state_dicts), optimizer/rng state.

## Reproduce

```bash
# build leakage-free train/val (171 clips + all variants excluded)
python backbone/scripts/build_w2v2_data.py --no-newspk
# fine-tune (v1 recipe, the active model):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/finetune_whisper.py \
    --out backbone/artifacts/whisper_base_ft
# eval on the held-out 171 clips:
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft.py \
    --model backbone/artifacts/whisper_base_ft/best \
    --data ../data/additional_test_data \
    --report backbone/reports/additional_test_whisper_ft.json
# in-domain regression guardrail (ME2 test split):
CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft_me2.py \
    --model backbone/artifacts/whisper_base_ft/best --n 400 --seed 42 \
    --report backbone/reports/me2_regression_whisper_ft.json
```
