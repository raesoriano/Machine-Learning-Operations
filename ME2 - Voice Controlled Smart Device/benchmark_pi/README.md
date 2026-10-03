# ME2 offline benchmark -- ONNX runner (Raspberry Pi)

Same 202-clip class holdout set and the same unmodified `vcmbench`
scoring code as `../benchmark`, but the current model (**v8new**:
causal Conformer + CTC, 736-word vocab, content-based reject) is
decoded with the **ONNX runtime** instead of torch. This is the
benchmark for machines that cannot install torch (Raspberry Pi 5):
the only Python dependencies are `onnxruntime` + `numpy` + the
scoring stack (`pandas`, `pyarrow`, `soundfile`).

The decode is byte-for-byte the same protocol as
`pi test v8-conformer-ctc/v8_onnx.py` (the Pi's live runtime):
constrained CTC-FSA decode over the 93 command phrases + greedy free
decode + content-based reject rule. The log-mel front-end is baked
into the ONNX graph, so there is no torchaudio anywhere.

## Run

```bash
cd "ME2 - Voice Controlled Smart Device/benchmark_pi"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python offline_bench_onnx.py
python make_report.py
```

The 202-clip holdout set auto-downloads to `/tmp/holdout.parquet` on
first run (internet needed once).

## Outputs (in `results/`)

| file | contents |
|---|---|
| `v8new_onnx_metrics.json` | all metrics (intent/command accuracy, false accept/reject, misfire, slot exact, latency) |
| `v8new_onnx_trials.csv` | per-clip: true vs predicted intent/slot, latency |
| `v8new_onnx_report.md` | benchmark-format report |
| `summary.json` | one-line-per-model summary |

## Notes

- The model files come from the sibling `pi test v8-conformer-ctc`
  folder: `models/best.onnx` + `models/best.onnx.data` +
  `models/meta.json` (all Git LFS -- run `git lfs pull` after a
  sparse checkout).
- Latency measured here is pure ONNX inference + CTC decode per clip
  (no wake word, no VAD, no file I/O), i.e. the same number the live
  demo prints as `asr_ms`. Expect it to be much higher than the
  DGX/A100 torch numbers (`../benchmark/results/v8new_dgx_*`).
- `../benchmark` keeps the torch runners (v3/v6/v7/v8/v8neg/v8new)
  for the cluster; this folder only needs the ONNX model, so it is
  safe to sparse-checkout just `benchmark_pi` +
  `pi test v8-conformer-ctc` on the Pi.
