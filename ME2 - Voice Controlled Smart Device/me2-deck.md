# ME2 — Voice Controlled Smart Device

*Always-on keyword + intent recognition at the edge · no cloud round-trip*

**3 October 2026** · Rowel Atienza (with OnIt, AI agent) · Model · Dataset · Training on A100 · Validation on RPi 5

---

## Model

```
Mic · 16 kHz → log-mel 80-dim (25 ms / 10 ms hop, baked into the ONNX)
Causal encoder · 6 layers · 256-dim · KV-cache · streaming (left-context only)
Intent head   · CTC 736 words + blank → constrained CTC-FSA over 93 phrases → 19 intents
Slot head     · slot value (time/percent/temperature/text) read off the decoded phrase
Actuator      · Raspberry Pi 5
```

| Item | Value |
|---|---|
| Architecture | causal 6-layer Conformer (d=256, nhead=4, d_ffn=1024) + word-level CTC head |
| Parameters / weights | **10.9 M · 43 MB** (ONNX, external-data; fp32) |
| Vocabulary | **736 words** (109 base command/number words + 627 out-of-scope words) + 1 CTC blank |
| Rejection | **content rule** — OOS speech decodes to real words that match none of the 93 command phrases → REJECT (not force-mapped) |
| Runtime | `onnxruntime` + `numpy` only (no torch / torchaudio on the Pi) |

## Validation on the Raspberry Pi 5

Benchmark = 202-clip class holdout (93 variations × 2 + 16 out-of-scope), scored
by the [vcm-benchmark](https://github.com/airimonda/vcm-benchmark) (offline
adaptation, unmodified scoring code). Accuracy is hardware-independent; latency
below is the **A100-cluster** decode — the RPi 5 run (same model, `onnxruntime`)
is pending the local run and will be slower (CPU).

| Item | Value |
|---|---|
| Keyword / intent acc | **85.1 %** intent (19) / **81.7 %** command (93) |
| Slot exact (intent correct) | **93.1 %** |
| False-accept rate | **87.5 %** (14/16 OOS fired a command) — the honest weak spot; see note |
| False-reject rate | **2.2 %** (in-scope clips rejected) |
| Misfire (wrong command) | **10.2 %** |
| Latency p50 / p95 | **42 / 73 ms** (A100 decode) · RTF ≪ 1 (streaming) |
| Runtime | `onnxruntime` · 1 thread |

> **Held-out test split (4,443 clips, 121 unseen speakers):** overall **90.0 %** ·
> command **90.8 %** · intent **91.8 %** · real-voice command **52.0 %** (synthetic
> 99.3 % — the synthetic-vs-real domain gap that limits every ME2 recognizer).
> The expanded 736-word vocab + OOS training cut real false-rejects from 15.2 % →
> **11.3 %**. The 87.5 % false-accept on the 16-clip OOS holdout is the trade-off:
> OOS now decodes to *real words* (so it is no longer force-rejected), and ~half of
> those contain base command words.

## Dataset

| Item | Value |
|---|---|
| Source | Hugging Face [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) |
| Hours / utts | **~30.9 h / 81,818** (train 10,733 · test 4,443 · holdout 202 · numerals 66,390 · synthetic_negatives 1,250) |
| Speakers | **315** (train) · **121** (test, **zero overlap with train**) |
| Real vs synthetic | test is ~81 % synthetic → real and synthetic scores reported separately |
| Labels | **19 intents** + OUT_OF_SCOPE · **18 slot values** (times, percents, temperatures, reminder texts) · 93 phrase variations |
| Sources | Speech Commands v2, MLEnd numerals, group synthetic TTS, SLURP, group real voices, SNIPS SLU, Fluent Speech Commands, Common Voice en, Xela S1–S5, Timers and Such, Xela Multi-Sensor (each keeps its own license) |

## Training on the A100 cluster

| Item | Value |
|---|---|
| Cluster | **3× A100** (of an 8× A100 DGX node), bf16 autocast |
| Objective | **CTC** (word-level, blank=0; `torch.nn.CTCLoss`) |
| Optimiser | **AdamW** (lr 1e-3, weight-decay 0.01, cosine schedule + 300-step warmup, grad-clip 5.0) |
| Steps / loss | **40 epochs** (per-GPU batch 48) · best val CTC loss **0.4870** · **~11 min wall** |
| Training set | 10,733 in-scope + 20,000 numerals + OOS train clips (real word targets) + 1,000 synthetic negatives (all-blank) ≈ 31.7k clips |
| Seed | 0 |

---

## To Be Submitted

| Item | Value |
|---|---|
| GitHub repository | [`raesoriano/Machine-Learning-Operations`](https://github.com/raesoriano/Machine-Learning-Operations) — public · `ME2 - Voice Controlled Smart Device/` |
| Dataset location | Hugging Face `airimonda/ai231-me2-voice-commands` (per-source licenses on the card; DOI pending) |
| A100 cluster | DGX node (8× A100), 3 GPUs used, 40 epochs ≈ 11 min, seed 0 |
| Model weights | in-repo via Git-LFS: `pi test v8-conformer-ctc/models/best.pt` + `best.onnx` (+ `.data`) |

## What the reviewer will look for

| # | Item | Status |
|---|---|---|
| 1 | Repo public, one-command reproduction | **done** — README has a one-go block (clone → download → train → eval → export → benchmark) |
| 2 | Dataset licensed and citable (DOI) | **partial** — HF card lists per-source licenses (SLURP CC-BY-4.0, Speech Commands CC-BY-4.0, …); a single DOI is pending |
| 3 | Training logs + final checkpoint committed | **done** — `models/best.pt` + `best.onnx` committed (LFS); eval JSONs in `pi test v8-conformer-ctc/` |
| 4 | RPi 5 latency reproduced by the posted script | **pending** — `v8_onnx.py` + `live_demo/vcm_live.py` posted; A100 decode is 73 ms p95, RPi 5 local run outstanding |
| 5 | Held-out test set, unseen speakers | **done** — 4,443 clips, 121 speakers, zero overlap with train |
| 6 | Baseline of comparable size compared | **done** — v3 (PocketSphinx 8 MB), v6 (HMM/GMM 0.14 MB), v7 (PocketSphinx retrained) all benchmarked side-by-side in `benchmark/results/comparison.md`; v8 is the largest (43 MB) and best |
