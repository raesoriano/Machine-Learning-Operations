# archive/ctc_v8 — frozen snapshot of the CTC pipeline (v1–v8)

This directory is an **unmodified snapshot** of the original VCM-v2 pipeline:
the small from-scratch CTC encoder (`me2_v6` / `me2_v8`), the two-stage
ASR→classifier design, and every report produced with it. It is kept as-is
for reproducibility and comparison. **All further development happens in
`../backbone/`** (pretrained Whisper base.en backbone).

## What is here

| Path | Contents |
|---|---|
| `vcm2/` | Pipeline code: `asr.py` (CTC ASR wrapper over ME2 checkpoints), `classifier.py` (TF-IDF+LogReg 31-command classifier), `normalize.py`, `ground_truth.py`, `pipeline.py`, `data_clean.py` |
| `scripts/` | `build_subset.py` (171-clip → 1539-variant training subset), `train_asr_v8.py` (fine-tune from me2_v6), `train_classifier.py`, `tune_classifier.py`, `eval.py` (end-to-end eval on `additional_test_data`), `eval_asr_split.py`, `eval_asr_variants.py`, `noise_probe.py` / `noise_ab.py`, `diagnose_v6_v8.py`, `ab_validate.py`, `typo_probe.py`, `classifier_only.py` |
| `reports/` | All result JSONs (see "Results" below) |
| `artifacts/` | `classifier.pkl` (final tuned classifier), `classifier_train.json`, `asr_me2_v8/best.pt` + `last.pt` + `config.yaml` (the me2_v8 fine-tuned CTC checkpoints, ~3.4 MB) |
| `data_new/` | `subset_manifest.csv` + `build_report.json` for the 171-clip new-speaker subset. The 1539 variant WAVs and the merged feature npz are **regenerable** (`scripts/build_subset.py`) and git-ignored. |

## Headline results (this pipeline)

Test set: `additional_test_data` — 171 clips, 19 intents, **one new speaker
held out of all training** (raw clips never seen in any form).

| ASR | Command (31-way) | Intent (19-way) | Blank | WER |
|---|---|---|---|---|
| me2_v6 (speaker-aug CTC, 422k params) | 24.6% | 28.1% | 38.6% | 76.5% |
| me2_v8 (v6 + 171-clip subset fine-tune) | 24.0% | 28.1% | 38.6% | 76.3% |

- **Classifier alone** (gold transcripts in): **99.4%** command accuracy —
  the text stage is at its ceiling.
- **Root cause of the gap:** the 422k-param CTC **underfits even its own
  in-domain data** (47.7% WER on the ME2 test split, 41.4% deletion) — a
  capacity deficit, not a speaker-coverage deficit. Adding the new speaker's
  clips (me2_v8) produced a null: 154/171 transcripts byte-identical to v6.
- **Pre-existing vocab defect (affects v6 and v8 equally):** the constrained
  1033-word vocab is missing `twenty`, `two`, `hundred`, so TEMPERATURE_22/26
  and BRIGHTNESS_100 targets silently lose their number.

## How to reproduce

```bash
# from the repo root
python archive/ctc_v8/scripts/build_subset.py --src ../additional_test_data
python archive/ctc_v8/scripts/train_asr_v8.py --features archive/ctc_v8/data_new/features \
    --init  "../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/best.pt" \
    --config "../Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/config.yaml" \
    --out archive/ctc_v8/artifacts/asr_me2_v8
python archive/ctc_v8/scripts/train_classifier.py --clean
python archive/ctc_v8/scripts/eval.py --data ../additional_test_data --report reports/additional_test_v8.json
```
