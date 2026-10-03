# ME2 benchmark

Offline adaptation of the [vcm-benchmark](https://github.com/airimonda/vcm-benchmark)
for the ME2 models. It reproduces the **same metrics** the live benchmark
produces, but headless: instead of the laptop→Pi SSH link, each model decodes
the identical 202-clip class holdout set with its **own production recognizer**
(the same code path the RPi runs), and the output is scored by the benchmark's
**unmodified** scoring code (`vcmbench/report.py`, `metrics.py`, `schema.py`,
`slots.py`).

The only metric that needs the live link — **false-wake rate** (commands played
*without* a wake word) — is reported as N/A offline.

## Test set

`airimonda/ai231-me2-voice-commands`, `split=holdout` — **202 clips**:
93 command variations × 2 voices + 16 out-of-scope. 96 real / 106 synthetic.
Downloaded on first run to `/tmp/holdout.parquet` (auto-cached).

## Models scored

| key | model | recognizer |
|---|---|---|
| `v3` | PocketSphinx ensemble (original AM) | `archived/pi test v3` |
| `v6` | from-scratch HMM/GMM | `archived/pi test v6` |
| `v7` | PocketSphinx ensemble (AM retrained on v6 data) | `archived/pi test v7` |
| `v8` | Conformer+CTC, 109-word vocab | `pi test v8-conformer-ctc/models_neg/best.pt` |
| `v8neg` | same, `--reject-empty` rule | `pi test v8-conformer-ctc/models_neg/best.pt` |
| `v8new` | Conformer+CTC, **736-word vocab + content reject** (current) | `pi test v8-conformer-ctc/models/best.pt` |

## Setup

```bash
cd "ME2 - Voice Controlled Smart Device/benchmark"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# to also score v3/v7, uncomment pocketsphinx + scikit-learn in requirements.txt
```

Paths are **relative to this folder** (the ME2 base dir is its parent), so it
runs from a full clone or a sparse checkout that includes `benchmark/`,
`pi test v8-conformer-ctc/`, and `archived/` (for v3/v6/v7).

## Run

```bash
# everything (v6 is the slow one: ~8.7 s/clip on CPU, ~30 min)
python offline_bench.py v3 v6 v7 v8 v8neg v8new

# just the current model (fast, ~1 min on GPU)
python offline_bench.py v8new
```

Writes per-model `<name>_metrics.json` + `<name>_trials.csv` into `results/`.

## Reports + comparison

```bash
python make_reports.py v3 v6 v7 v8 v8neg v8new   # -> results/<name>_report.md
python make_comparison.py                          # -> results/comparison.md + .json
```

`results/comparison.md` is the side-by-side table (intent/command accuracy,
false-accept, false-reject, misfire, slot exact, latency, real vs synthetic).

## Committed results

`results/` holds the **DGX-cluster run** (8× A100, 2026-10-03). Headline
(`v8new`, the current model): **intent 85.1 % · command 81.7 % · slot exact
93.1 % · false-accept 87.5 % (14/16) · false-reject 2.2 % · misfire 10.2 %**.
To reproduce locally on your RPi, run the same `offline_bench.py` commands
above (v8/v8new run on CPU fine; v6 is slow on the Pi).

## Metrics produced (per model)

- **intent level** (19 intents): accuracy + 95 % CI, balanced accuracy, macro
  precision/recall/F1/F2, false-accept rate (OOS clips that fired a command),
  false-reject rate (in-scope clips rejected), misfire rate, per-class table.
- **command level** (93 variations): same set of metrics.
- **slot exact**: fraction of intent-correct clips whose slot value is exact.
- **breakdowns**: overall / real voice / synthetic voice.
- **latency**: p50 / p95 decode time per clip.
