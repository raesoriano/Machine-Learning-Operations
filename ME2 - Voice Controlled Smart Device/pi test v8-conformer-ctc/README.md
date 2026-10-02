# pi test v8-conformer-ctc — ME2 voice command recognizer: **neural causal encoder (original template)**

The **original ME2 template architecture** — a causal (left-context only)
encoder with a word-level **CTC** head — trained from scratch on the AI231
ME2 **v6 dataset** (updated 2026-10-02) on the DGX box (8× A100). No
pre-trained acoustic model, no PocketSphinx: pure torch.

## Architecture

```
16 kHz waveform
  -> 80-dim log-mel (torchaudio, 25 ms / 10 ms)
  -> conv front-end (stride-4 stack)          [B, T/4, 256]
  -> 6-layer causal Conformer (d_model=256, nhead=4, d_ffn=1024)
  -> CTC head: 736 words + 1 blank
```

Causality: every block uses only left context (causal self-attention with a
lower-triangular mask; convolutions are left-padded; FFN is pointwise), so the
encoder can stream frame-by-frame. **10.9 M parameters.**

## Vocabulary (expanded — the rejection fix)

The current model uses a **736-word vocabulary**:

* **words 1–109** — the base v6 dictionary (in-scope command words + number
  words). Indices are stable across versions.
* **words 110–736** — every word that appears in the **out-of-scope (OOS)
  train transcripts** (627 words), each with a placeholder phone (`SIL`); the
  CTC head learns their acoustic emission directly from the OOS training
  clips.

**Why:** with the base 109-word vocab, OOS speech could not be decoded at all
(free decode came out empty/garbage), so the reject rule had nothing to work
with. With the expanded vocab the model **decodes OOS speech to the real
words**; rejection then happens at the command-matching stage, where those
words do not form any of the 93 in-scope command phrases. This is what fixes
the false rejections: in-scope commands still decode to a high-scoring
constrained phrase, OOS decodes to words that match no command.

## Data (the v6 dataset, updated)

* **in-scope train clips** (10,733) + **numerals split** (20,000, real
  speech) + **OOS train clips** (their real word targets) + **1,000 synthetic
  negatives** (all-blank targets) = ~31.7k training clips (16 kHz mono).
* word vocab = 736 words as above; word 0 in the CTC output is the blank.
* validation: the 202-clip holdout split (in-scope + OOS).

## Training (DGX)

```
torchrun --nproc_per_node=3 train_v8.py --epochs 40 --bs 48 --workers 8 \
  --out models \
  --negatives /path/to/me2-v6-negatives/dataset
```

* 3× A100, bf16 autocast, AdamW (lr 1e-3, cosine + warmup), grad clip 5.0.
  **40 epochs in ~11 min.**
* best checkpoint by validation CTC loss: **`models/best.pt`, val_loss
  0.4870.**

## Decoding

* **constrained decode** : CTC-FSA dynamic program over the 93 command
  phrases (uniform phrase priors).
* **free decode**        : greedy CTC (collapse repeats, drop blanks).
* **REJECT — content rule** (`--reject-content`, the default for this
  model): the free decode does **not** look like a command, i.e. either
  * (a) fewer than 60% of its words are base-109 command words, or
  * (b) the FSA locked a 1-word command onto a multi-word utterance
    (`phrase_len / free_len < 0.20`).
  (Empty free decode ⇒ base-frac 0 ⇒ rejected.)
* **REJECT — legacy rules** (for the old 109-vocab models):
  score-gap `(free_lp - constrained_lp)/frames > margin` and
  `--reject-empty` (free decode empty).
* **gold**               : manifest `command` column (coarse 19 schema);
  OUT_OF_SCOPE rows are the REJECT class.

```
python eval_v8.py --split test --reject-content --report _eval_test.json
```

### Speed: multi-GPU sharding

The full 4,443-clip test runs in **~70 s wall on 3 GPUs**:

```
python eval_v8.py --split test --reject-content --num-shards 3 --shard-id 0 --device cuda:0 --report _eval_test.json &
python eval_v8.py --split test --reject-content --num-shards 3 --shard-id 1 --device cuda:1 --report _eval_test.json &
python eval_v8.py --split test --reject-content --num-shards 3 --shard-id 2 --device cuda:2 --report _eval_test.json &
wait
python eval_v8.py --split test --reject-content --merge --num-shards 3 --report _eval_test.json
```

Mel features are computed in a CPU thread pool overlapped with GPU encoding.
`--selftest` cross-checks the FSA decoder against a pure-Python oracle
(120/120 identical).

## ONNX / standalone runtime (no torch, no torchaudio)

The model is exported to ONNX with the log-mel front-end **baked into the
graph** (input: 16 kHz float32 waveform `[B, T]` → output: CTC log-probs
`[B, T', 737]`), so the runtime needs only `onnxruntime` + `numpy` + the
stdlib — no torch, no torchaudio, and no import from `pi test v6` (the vocab
and the 93 command phrases are baked into `meta.json` next to the ONNX file).

```
python export_onnx.py --model models/best.pt --out models/best.onnx
```

Exported with the dynamo exporter at opset 18 (the legacy TorchScript
exporter cannot trace `torch.stft`). Weights are stored in the sibling
`.onnx.data` external-data file (both tracked by LFS).

```
# Pi-side: pip install onnxruntime numpy
python v8_onnx.py --model models/best.onnx --file clip.wav --reject-content
python v8_onnx.py --model models/best.onnx --data <dataset> --split test --reject-content --report report.json
```

`v8_onnx.py` runs the exact same decode protocol as `eval_v8.py`
(constrained CTC-FSA + greedy free decode + reject rule). Verified after
export on holdout clips against the fp32 torch model: **zero per-clip
argmax flips** (max |Δlog-prob| 2.3e-5).

| model | ONNX | runtime |
|---|---|---|
| **v8 (736-vocab, current)** | `models/best.onnx` (+`.data`, 43 MB) | `v8_onnx.py --model models/best.onnx --reject-content` |
| v8 + negatives (old, 109-vocab) | `models_neg/best.onnx` (+`.data`, 43 MB) | `v8_onnx.py --model models_neg/best.onnx --reject-empty` |

## Results (updated v6 test split, 4,443 clips — 4,367 in-scope + 76 OOS)

| metric | old (109-vocab, reject-empty) | **current (736-vocab, reject-content)** |
|---|---:|---:|
| **overall accuracy** | 0.8888 | **0.9001** |
| command accuracy | 0.8949 | **0.9084** |
| intent accuracy | 0.9037 | **0.9183** |
| **real false-reject** | 0.152 | **0.113** |
| reject accuracy (OOS) | 0.5395 | 0.4211 |
| decode latency p50 / p90 | 25.5 / 41.7 ms | 24.5 / 40.4 ms |

**Real vs synthetic** (test is 81% synthetic):

| | n | overall | command | reject |
|---|---|---|---|---|
| real | 824 | 0.5158 | 0.5199 | 0.4468 |
| synthetic | 3,619 | 0.9876 | 0.9925 | 0.3793 |

**Synthetic negatives split (250 clips):** reject **0.824** (old model:
0.892).

### Reading the numbers

* **Synthetic 0.99** — the encoder essentially memorizes the synthetic
  command voices (the majority of the training data).
* **Real 0.52** — the synthetic-vs-real domain gap that limited every ME2
  recognizer; the expanded vocab + OOS training lifted real command accuracy
  from 0.44 to 0.52 and cut real false-rejects from 15.2% to 11.3%.
* **OOS reject 0.42** — the honest trade-off of the vocab expansion: OOS
  speech now decodes to *real words*, and ~half of those contain base command
  words ("pause the", "what time is it"), so the content rule accepts them.
  The old model rejected more OOS (0.54) but by emitting *nothing* for real
  commands too (15.2% false-reject).

## Files

| Path | What it is |
|---|---|
| `model.py` | causal Conformer + CTC encoder (10.9 M params) |
| `data.py` | 16 kHz wav → log-mel, tokenizer, CTC targets, `build_vocab_v8` |
| `train_v8.py` | DDP training (bf16, cosine LR, val on holdout) |
| `eval_v8.py` | CTC-FSA constrained decode + free decode + reject rules, metrics, sharding |
| `calibrate_reject.py` | sweep reject-rule thresholds on a merged eval report |
| `analyze_reject.py` | per-clip score-distribution analysis (rule design) |
| `export_onnx.py` | export the full graph (wav → log-probs) to ONNX + bake `meta.json` |
| `v8_onnx.py` | **standalone ONNX runtime** (onnxruntime + numpy only, no torch) |
| `requirements_onnx.txt` | Pi-side deps for `v8_onnx.py` (onnxruntime + numpy) |
| `base_dictionary.txt` | the base 109-word v6 dictionary (phones) |
| `variations.csv` | the 93 command phrase variations |
| `models/best.pt` | best checkpoint (736-vocab, val_loss 0.4870) |
| `models/best.onnx` (+`.data`) | same model, ONNX (opset 18, front-end baked in) |
| `models/meta.json` | baked 736-word vocab + 93 command phrases |
| `models/words.txt` | the 736-word vocab (base 109 first) |
| `models_neg/` | old 109-vocab negatives-trained model (kept for reference) |
| `_eval_final_test.json` | full per-clip + per-class test report (current model) |
| `_eval_final_negtest.json` | 250-clip synthetic negatives report |
