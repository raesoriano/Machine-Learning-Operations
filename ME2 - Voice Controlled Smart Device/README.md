# ME2 — Voice-Controlled Smart Device

A **tiny voice-command model (VCM)** for a smart device: it understands the
**19 command intents** people give smart devices (play music, weather, time,
lights, brightness, color, timer, alarm, temperature, media control, call,
message, reminders), runs **on-device in real time** on a **Raspberry Pi 5**,
fully **standalone** (no cloud, no LLM). Unknown speech is **rejected**, not
guessed.

The current model is **`pi test v8-conformer-ctc`** — a **causal Conformer +
CTC** encoder (the original ME2 template architecture), **10.9 M parameters**,
exported to **ONNX** so the Pi runs it with only `onnxruntime` + `numpy`.

```
Mic · 16 kHz → log-mel (baked into the ONNX)
  → causal 6-layer Conformer (d=256, streaming, KV-cache)
  → CTC head (736 words + blank)
  → constrained CTC-FSA decode (93 command phrases) + free decode
  → content-based REJECT (out-of-scope decodes to real words that match no command)
  → 19 intents + 18 slots
  → actuator (Raspberry Pi 5)
```

**Headline results** (held-out `test`, 4,443 clips, 121 unseen speakers):
**90.0 % overall · 90.8 % command · 91.8 % intent**, decode latency **p50 24.5 ms**.
Benchmark (202-clip class holdout): **85.1 % intent · 81.7 % command · 93.1 % slot exact**.
Full tables in [`pi test v8-conformer-ctc/README.md`](pi%20test%20v8-conformer-ctc/README.md)
and [`benchmark/results/comparison.md`](benchmark/results/comparison.md).

---

## Repository layout

```
ME2 - Voice Controlled Smart Device/
├── README.md                     # this file
├── me2-deck.md                   # results deck (Dr. Atienza template)
├── pi test v8-conformer-ctc/     # *** CURRENT MODEL *** — training, eval, ONNX, runtime
│   ├── train_v8.py  eval_v8.py  export_onnx.py  v8_onnx.py
│   ├── model.py  data.py  hgm/  base_dictionary.txt  variations.csv
│   └── models/  models_neg/      # checkpoints + ONNX (Git-LFS)
├── data/                         # dataset description + download script (audio git-ignored)
│   ├── README.md  download_dataset.py
├── benchmark/                    # vcm-benchmark (offline) + cluster results
│   ├── offline_bench.py  make_reports.py  make_comparison.py  vcmbench/
│   └── results/                  # committed DGX-cluster run
├── live_demo/                    # reusable live-mic demo (wake word → command → speak)
│   ├── vcm_live.py  shared.py  adapters/  responses/  wakeword/
└── archived/                     # older models (v3–v7) + legacy data — see below
```

## Getting the code (sparse checkout)

The repo is large (model weights are Git-LFS objects). Check out only what you
need. **LFS must be installed first** or the weights arrive as 130-byte pointers.

```bash
sudo apt update
sudo apt install -y git git-lfs python3-venv
git lfs install

git clone --filter=blob:none --no-checkout \
  https://github.com/raesoriano/Machine-Learning-Operations.git ml-ops
cd ml-ops
# pick the folders you need:
git sparse-checkout set \
  "ME2 - Voice Controlled Smart Device/pi test v8-conformer-ctc" \
  "ME2 - Voice Controlled Smart Device/data" \
  "ME2 - Voice Controlled Smart Device/benchmark" \
  "ME2 - Voice Controlled Smart Device/live_demo"
git checkout main
cd "ME2 - Voice Controlled Smart Device"
```

| I want to… | sparse-checkout these |
|---|---|
| **Train / eval** the v8 model | `pi test v8-conformer-ctc`, `data` |
| **Run the benchmark** | `benchmark`, `pi test v8-conformer-ctc`, `data`, `archived` (for v3/v6/v7) |
| **Run the live demo** (v8) | `live_demo`, `pi test v8-conformer-ctc` |
| **Everything** | omit `git sparse-checkout set` (full checkout) |

Verify the weights downloaded (each should be ~43 MB, **not** ~130 bytes):
`ls -la "pi test v8-conformer-ctc/models/best.onnx"`. If it's tiny, run
`git lfs pull` in `ml-ops`.

---

## The dataset

See [`data/README.md`](data/README.md) for the full description. In short:
**[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)**
— 81,818 clips / ~30.9 h, 19 intents + 18 slots + out-of-scope. Splits:
`train` 10,733 · `test` 4,443 (121 unseen speakers) · `holdout` 202 ·
`numerals` 66,390 · `synthetic_negatives` 1,250.

Download it (one command, ~7 GB):

```bash
cd "ME2 - Voice Controlled Smart Device/data"
pip install datasets pandas soundfile
python download_dataset.py --negatives     # -> ./dataset/{train,test,holdout,numerals,synthetic_negatives}
```

The v8 code's default `--data` points at `data/dataset/` (created by the
script). Pass `--data <path>` to override.

---

## Training the v8 model (DGX / A100 cluster)

The model was trained on **3× A100** in ~11 min (40 epochs). The same commands
work on any multi-GPU box; on a single GPU drop `torchrun` and run
`python train_v8.py`.

```bash
cd "ME2 - Voice Controlled Smart Device"
python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio numpy

cd "pi test v8-conformer-ctc"
# (dataset must already be downloaded into ../data/dataset, see above)
torchrun --nproc_per_node=3 train_v8.py --epochs 40 --bs 48 --workers 8 \
  --out models \
  --negatives ../data/dataset/synthetic_negatives
```

Outputs `models/best.pt` (best by validation CTC loss) + `models/words.txt`.
bf16 autocast, AdamW (lr 1e-3, cosine + warmup), grad-clip 5.0.

## Evaluating the v8 model

```bash
cd "ME2 - Voice Controlled Smart Device/pi test v8-conformer-ctc"
# single GPU (~15 min for the 4,443-clip test split):
python eval_v8.py --split test --reject-content --report _eval_test.json
# multi-GPU sharded (3× A100, ~70 s wall):
python eval_v8.py --split test --reject-content --num-shards 3 --shard-id 0 --device cuda:0 --report _eval_test.json &
python eval_v8.py --split test --reject-content --num-shards 3 --shard-id 1 --device cuda:1 --report _eval_test.json &
python eval_v8.py --split test --reject-content --num-shards 3 --shard-id 2 --device cuda:2 --report _eval_test.json &
wait
python eval_v8.py --split test --reject-content --merge --num-shards 3 --report _eval_test.json
```

`--reject-content` is the current model's reject rule (see the v8 README for
`--reject-empty`, the legacy rule).

## Exporting to ONNX (standalone runtime)

```bash
cd "ME2 - Voice Controlled Smart Device/pi test v8-conformer-ctc"
python export_onnx.py --model models/best.pt --out models/best.onnx
```

Bakes the log-mel front-end + vocab + 93 phrases into the ONNX graph, so the
Pi runtime needs **only `onnxruntime` + `numpy`** (no torch/torchaudio).
Verified: zero per-clip argmax flips vs the fp32 torch model.

---

## Running the benchmark

The [vcm-benchmark](https://github.com/airimonda/vcm-benchmark) adapted to run
headless — same metrics, no laptop→Pi SSH link. Full instructions in
[`benchmark/README.md`](benchmark/README.md).

```bash
cd "ME2 - Voice Controlled Smart Device/benchmark"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python offline_bench.py v8new          # current model (~1 min on GPU)
# all models (v6 is slow on CPU, ~30 min):
python offline_bench.py v3 v6 v7 v8 v8neg v8new
python make_reports.py v3 v6 v7 v8 v8neg v8new
python make_comparison.py              # -> results/comparison.md
```

Committed cluster results are in `benchmark/results/`.

---

## Live demo (Raspberry Pi 5)

`live_demo/` is the reusable live-mic demo: **wake word → "yes?" cue → command
→ spoken response → standby**, with a **swappable model layer** so you can run
v8 (current), v3, or v7 from one script. Full instructions in
[`live_demo/README.md`](live_demo/README.md).

**v8 on the Pi (lightest — no torch):**

```bash
cd "ME2 - Voice Controlled Smart Device/live_demo"
python3 -m venv .venv && source .venv/bin/activate
pip install onnxruntime numpy sounddevice soundfile scipy webrtcvad-wheels \
            openwakeword==0.4.0 piper-tts yt-dlp
sudo apt install -y mpv                 # for the music commands

python vcm_live.py --model v8           # LIVE MIC: "hey rhasspy" -> command -> speak
python vcm_live.py --model v8 --file clip.wav   # classify one file
```

Other models: `--model v3` / `--model v7` (need `pocketsphinx` +
`scikit-learn` uncommented in `live_demo/requirements.txt`).

---

## One-go (everything, on the A100 cluster)

```bash
sudo apt install -y git git-lfs python3-venv
git lfs install
git clone --filter=blob:none --no-checkout \
  https://github.com/raesoriano/Machine-Learning-Operations.git ml-ops
cd ml-ops
git sparse-checkout set \
  "ME2 - Voice Controlled Smart Device/pi test v8-conformer-ctc" \
  "ME2 - Voice Controlled Smart Device/data" \
  "ME2 - Voice Controlled Smart Device/benchmark"
git checkout main
cd "ME2 - Voice Controlled Smart Device"

python3 -m venv .venv && source .venv/bin/activate
pip install torch torchaudio numpy datasets pandas soundfile

# 1. dataset
(cd data && python download_dataset.py --negatives)

# 2. train
(cd "pi test v8-conformer-ctc" && \
  torchrun --nproc_per_node=3 train_v8.py --epochs 40 --bs 48 --workers 8 \
    --out models --negatives ../data/dataset/synthetic_negatives)

# 3. eval
(cd "pi test v8-conformer-ctc" && \
  python eval_v8.py --split test --reject-content --report _eval_test.json)

# 4. export ONNX
(cd "pi test v8-conformer-ctc" && \
  python export_onnx.py --model models/best.pt --out models/best.onnx)

# 5. benchmark
(cd benchmark && pip install -r requirements.txt && \
  python offline_bench.py v8new && python make_reports.py v8new && python make_comparison.py)
```

---

## Other models tried (in `archived/`)

ME2 went through several architectures before settling on v8. All are kept in
[`archived/`](archived/) for reference and as baselines:

| model | architecture | notes |
|---|---|---|
| **v3** (`archived/pi test v3`) | wake-word-gated **PocketSphinx ensemble** (custom + stock AM, stage-2 classifier) | first working live demo; 31 fine classes |
| **v4** (`archived/pi test v4`) | PocketSphinx, reports what was heard + rejects non-commands | diagnostic iteration |
| **v5** (`archived/pi test v5`) | lightweight **ONNX** listener | early ONNX attempt |
| **v6** (`archived/pi test v6`) | from-scratch **HMM/GMM** (bootstrap EM, CPU) | the original template arch; 109 words |
| **v7** (`archived/pi test v7`) | **PocketSphinx ensemble**, AM retrained on the v6 dataset | v3 arch on the new data |
| **v8** (`pi test v8-conformer-ctc`) | **causal Conformer + CTC** (current) | 736-word vocab, content reject, ONNX |

`archived/` also holds the legacy v5-era data build (`archived/data-legacy/`)
and earlier planning/report docs. The v8 model strictly beats every archived
model on the held-out test split (see the benchmark comparison).
