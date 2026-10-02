# pi test v8-conformer-ctc — ME2 voice command recognizer: **neural causal encoder (original template)**

The **original ME2 template architecture** — a causal (left-context only)
encoder with a word-level **CTC** head — trained from scratch on the AI231
ME2 **v6 dataset** on the DGX box (8× A100). No pre-trained acoustic model,
no PocketSphinx: pure torch.

## Architecture

```
16 kHz waveform
  -> 80-dim log-mel (torchaudio, 25 ms / 10 ms)
  -> conv front-end (stride-4 stack)          [B, T/4, 256]
  -> 6-layer causal Conformer (d_model=256, nhead=4, d_ffn=1024)
  -> CTC head: 109 words + 1 blank
```

Causality: every block uses only left context (causal self-attention with a
lower-triangular mask; convolutions are left-padded; FFN is pointwise), so the
encoder can stream frame-by-frame. **10.7 M parameters.**

## Data (the v6 dataset)

Same clip selection as the v6 HMM/GMM run, so the models are directly
comparable:

* **in-scope train clips** (8,834) + **numerals split** (20,000, real speech)
  = 28,834 training clips (16 kHz mono).
* out-of-scope rows (201) and OOV transcripts (1,647) excluded.
* word vocab = the 109-word v6 dictionary (in-scope command words + number
  words); word 0 in the CTC output is the blank.

## Training (DGX)

```
torchrun --nproc_per_node=6 train_v8.py --epochs 12 --bs 48 --workers 8
```

* 6× A100, bf16 autocast, AdamW (lr 1e-3, cosine + 300-step warmup),
  grad clip 5.0. **12 epochs in ~95 s.**
* best checkpoint by validation CTC loss on the 196-clip holdout split:
  **best.pt, val_loss 0.4497.**

## Decoding (same protocol as v6)

* **constrained decode** : CTC-FSA dynamic program over the 93 command
  phrases (uniform phrase priors).
* **free decode**        : greedy CTC (collapse repeats, drop blanks).
* **REJECT**             : free decodes far better than the best grammar
  phrase, i.e. `(free_lp - constrained_lp)/frames > margin` (default 8.0).
* **gold**               : manifest `command` column (coarse 19 schema);
  OUT_OF_SCOPE rows are the REJECT class.

```
python eval_v8.py --split test --report _eval_test.json
```

## ONNX / standalone runtime (no torch, no torchaudio)

The model is exported to ONNX with the log-mel front-end **baked into the
graph** (input: 16 kHz float32 waveform `[B, T]` → output: CTC log-probs
`[B, T', 110]`), so the runtime needs only `onnxruntime` + `numpy` + the
stdlib — no torch, no torchaudio, and no import from `pi test v6` (the vocab
and the 93 command phrases are baked into `meta.json` next to the ONNX file).

```
python export_onnx.py --model models/best.pt --out models/best.onnx
python export_onnx.py --model models_neg/best.pt --out models_neg/best.onnx
```

Exported with the dynamo exporter at opset 18 (the legacy TorchScript
exporter cannot trace `torch.stft`). Weights are stored in the sibling
`.onnx.data` external-data file (both tracked by LFS).

```
# Pi-side: pip install onnxruntime numpy
python v8_onnx.py --model models/best.onnx --file clip.wav
python v8_onnx.py --model models_neg/best.onnx --file clip.wav --reject-empty
python v8_onnx.py --model models/best.onnx --data <dataset> --split test --report report.json
```

`v8_onnx.py` runs the exact same decode protocol as `eval_v8.py`
(constrained CTC-FSA + greedy free decode + reject rule). Verified on the
196-clip holdout split against the fp32 torch model: **zero per-clip
prediction flips** for both checkpoints (max |Δlog-prob| 1.1e-4 / 7.9e-5).
Note: `eval_v8.py` on a GPU uses bf16 autocast, which itself flips a small
number of near-tie decodes vs fp32 — the ONNX runtime matches the fp32
reference exactly.

| model | ONNX | runtime |
|---|---|---|
| base v8 | `models/best.onnx` (+`.data`, 43 MB) | `v8_onnx.py --model models/best.onnx` |
| v8 + negatives | `models_neg/best.onnx` (+`.data`, 43 MB) | `v8_onnx.py --model models_neg/best.onnx --reject-empty` |

(int8 dynamic quantization was attempted but the quantized graph does not
export cleanly with torch 2.14; the fp32 ONNX is the supported artifact.)

## Results (v6 test split, 4,418 clips — 4,371 in-scope + 47 out-of-scope)

| metric | value |
|---|---|
| **overall accuracy** | **0.8497** |
| command accuracy | 0.8588 |
| intent accuracy | 0.8717 |
| reject accuracy | 0.0000 |
| decode latency p50 / p90 | 24.9 / 43.0 ms |

**Real vs synthetic** (test is 77% synthetic):

| | n | overall | command | reject |
|---|---|---|---|---|
| real | 1,050 | 0.4019 | 0.4207 | 0.0000 |
| synthetic | 3,368 | 0.9893 | 0.9893 | 0.0000 |

### Reading the numbers

* **Synthetic 0.99** — the encoder essentially memorizes the synthetic
  command voices (the majority of the training data).
* **Real 0.40** — the same **synthetic-vs-real domain gap** that limited v5
  and v6: the model was trained on a synthetic-majority mix, so real human
  voices decode poorly. (v6's `--real-only` retrain attacked this for the
  HMM/GMM model; the analogous fix here would be a real-only retrain.)
* **Reject 0.0** — all 47 out-of-scope clips are mis-decoded as commands;
  the constrained-vs-free score gap never exceeds the margin. Rejection on
  this split is the weak point for every ME2 recognizer so far.
* **Overall 0.85** is in line with the v3 PocketSphinx ensemble baseline
  (0.848 command / 0.883 intent on its 171-clip set) and well above the v6
  HMM/GMM real-only run (0.429 overall on this same test split).

## Files

| Path | What it is |
|---|---|
| `model.py` | causal Conformer + CTC encoder (10.7 M params) |
| `data.py` | 16 kHz wav → log-mel, spoken-word tokenizer, CTC targets |
| `train_v8.py` | DDP training (bf16, cosine LR, val on holdout) |
| `eval_v8.py` | CTC-FSA constrained decode + free decode + reject, metrics |
| `export_onnx.py` | export the full graph (wav → log-probs) to ONNX + bake `meta.json` |
| `v8_onnx.py` | **standalone ONNX runtime** (onnxruntime + numpy only, no torch) |
| `requirements_onnx.txt` | Pi-side deps for `v8_onnx.py` (onnxruntime + numpy) |
| `models/best.pt` | best checkpoint (val_loss 0.4497) |
| `models/best.onnx` (+`.data`) | same model, ONNX (opset 18, front-end baked in) |
| `models/meta.json` | baked vocab + 93 command phrases (for `v8_onnx.py`) |
| `models/words.txt` | the 109-word vocab |
| `_eval_test.json` | full per-clip + per-class test report |
