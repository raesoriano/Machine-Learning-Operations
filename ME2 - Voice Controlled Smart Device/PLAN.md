# Tiny Voice Command Model (VCM) — Project Plan

**ME goal:** Build a tiny, standalone Voice Command Model (VCM) that understands the most
common commands humans issue to smart devices, and runs in real time on a Raspberry Pi 4/5
with **no cloud access and no LLM** — everything on-device.

**Why not a general ASR?** Full ASR models (Whisper-class and similar) are undesirable for
on-device use because of footprint: hundreds of MB of weights, high RAM/CPU cost, and
latency that is hard to fit on an RPi. A VCM is purpose-built: a small command vocabulary,
a fixed set of intents, and a few slot types (numbers, times, contacts, rooms). That
constraint is exactly what lets the model be tiny.

---

## 1. Command taxonomy (what the VCM must understand)

Ranked by real-world usage (259,164 logged Alexa + Google Home commands, plus recurring
U.S. consumer surveys). Each intent lists its slots (arguments the model must extract).

| # | Intent | Example utterances | Slots |
|---|--------|--------------------|-------|
| 1 | `play_music` | "play music", "play jazz", "play lofi" | `query` (optional: genre/artist) |
| 2 | `ask_question` | "what's the weather", "what time is it" | `topic` ∈ {weather, time} |
| 3 | `lights_switch` | "turn on the lights", "turn off the kitchen light" | `state` ∈ {on, off}, `location` (optional) |
| 4 | `lights_adjust` | "dim the lights to 40 percent", "set lights to blue" | `percent` (0–100) or `color` |
| 5 | `set_timer` | "set a timer for 10 minutes" | `duration` (minutes/hours) |
| 6 | `set_alarm` | "set an alarm for 6 am" | `time` (12-h clock, am/pm) |
| 7 | `set_temperature` | "set temperature to 22 degrees" | `temp` (°C/°F range) |
| 8 | `media_control` | "pause", "stop", "next", "skip", "volume up", "louder" | none (sub-intent is the command) |
| 9 | `reminders_lists` | "remind me to call mom at 5 pm", "what are my reminders" | `action`, `time` (optional) |
| 10 | `call` | "call mom" | `contact` |

Notes:
- Intent 1 ("play music") can be folded into the media/music family, but we keep it
  separate because it is the #1 use case in every survey year (2018–2020).
- Intent 3 accounts for ~85% of all Alexa IoT commands; intent 4 ~10%.
- Slot spaces are **closed and enumerable** (see §3.1) — this is what makes a tiny model
  feasible.

---

## 2. Architecture (individual work — items 2, 4, 6, 7)

### 2.1 Design options

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **A. Tiny end-to-end ASR with constrained decoding (recommended)** | Small RNN-T (1–5M params) or distilled small Conformer trained on the command dataset, with a restricted vocabulary (command words + digits + slot values). A deterministic parser maps the transcript to `(intent, slots)`. | Robust to paraphrase; scales to all 10 intents + slots; vocabulary constraint keeps it tiny; transcript is human-auditable | Needs careful slot-token design |
| B. Direct intent+slot classifier | Small TCN/CNN over log-mel (1–5M params) classifies intent; a small CTC digit head or second head emits slots. | Simplest, lowest latency | Rigid to paraphrase; each new slot type needs retraining |
| C. Keyword spotting + rule parser | SpeechCommands-style KWS per phrase + rules. | Extremely small | Fragile to variation; poor slot extraction; does not generalize |

**Decision:** Start with **Option A** (tiny RNN-T, constrained vocab). If the RPi4
budget is missed, fall back to Option B with the same data.

### 2.2 Pipeline (all on-device)

```
mic ──► VAD (webrtc/silero, ~1MB) ──► wake word (optional, openWakeWord/Porcupine, <5MB)
     ──► VCM inference (ONNX Runtime, int8) ──► transcript (constrained vocab)
     ──► deterministic parser ──► {intent, slots} JSON ──► device backend (local MQTT/HTTP)
```

- Wake word is optional (needed for the demo to be pleasant, not for correctness).
- Parser is pure code — no LLM, no cloud.

### 2.3 Hard constraints (acceptance criteria for "tiny")

| Metric | Target (must meet on **RPi4**, 4-core A72) | Stretch (RPi5) |
|--------|--------------------------------------------|----------------|
| Model size (int8) | ≤ 10 MB | ≤ 20 MB |
| Peak RAM (whole pipeline) | ≤ 250 MB | ≤ 300 MB |
| Real-time factor (RTF) | ≤ 0.5 (i.e., ≥ 2× real time) | ≤ 0.3 |
| End-to-end latency (speech end → command JSON) | p50 ≤ 500 ms, p95 ≤ 1 s | p50 ≤ 300 ms |
| CPU | ≤ 1.5 cores sustained (non-wake) | — |
| Network | **none at runtime** | none |

Runtime stack: Raspberry Pi OS 64-bit, Python 3.11, ONNX Runtime (or TFLite/NCNN) with
dynamic int8 quantization, 16 kHz mono input, 30 ms frames.

---

## 3. Dataset (collective work — item 1)

### 3.1 Slot spaces (closed, enumerable)

- `percent`: 5, 10, …, 100 (spoken: "forty percent", plus "half", "a little")
- `duration`: 1–120 minutes, 1–24 hours
- `time`: 12-hour clock, all hours × {am, pm}
- `temp`: 16–30 °C (and 60–85 °F variants)
- `contact`: fixed list (mom, dad, sister, brother, friend, … ~20 names)
- `location`: living room, kitchen, bedroom, bathroom, office, hallway
- `color`: red, blue, green, yellow, white, warm, cool
- `music query`: "music", genres (rock, jazz, pop, classical, lofi, country), ~20 artist names

### 3.2 Utterance generation

1. **Template grammars** per intent, e.g. for `lights_adjust`:
   - "dim the {location?} lights to {percent} percent"
   - "set the {location?} lights to {color}"
   - "make the lights {percent} percent bright"
   Each template expands over its slot space → thousands of unique utterances per intent.
2. **TTS (offline, at data-generation time only — never at runtime):**
   - Piper (many voices, fast, fully local) as the workhorse;
   - 1–2 additional open TTS engines (e.g., Coqui TTS) for voice diversity;
   - target **20–40 distinct voices** with accent diversity (US, UK, PH, IN).
3. **Target scale:** ~50k–100k training utterances (TTS is cheap); minimum ~2k utterances
   per intent, with slot coverage balanced (every slot value appears ≥ 20×).
4. **Augmentation:**
   - Spec augmentation (time/frequency masking)
   - Convolution with real RIRs (OpenSLR room impulse responses) → far-field simulation
   - Additive noise (MUSAN / NOISEI) at SNR 5–25 dB
   - Small speed/pitch jitter (±5%)
5. **Real-data subset (domain-gap check):** record 100–500 utterances from 5–10 people
   with a phone/USB mic; use only for validation, never in training.
6. **Labels:** auto-generated from templates → gold `(intent, slots)` JSON per utterance.
   No manual labeling needed for the core set; spot-check 500 samples for QA.
7. **Reproducibility:** fixed seeds, generation scripts in `data/generate/`, dataset card
   in `data/DATASET_CARD.md` (sources, licenses, scale, splits, known biases).

### 3.3 Splits

- Train / Val / Test = 80 / 10 / 10, **split by speaker** (no voice leakage).
- The test set doubles as the **frozen benchmark set** (§4): ~2,000 utterances
  (≈200 per intent × 10 held-out speakers) + 500 OOV/rejection utterances +
  a noise-robustness subset (SNR 20/10/5 dB).

---

## 4. Benchmark (collective work — item 3)

A frozen, versioned test set + a standard evaluation harness so every team's VCM is
compared under identical conditions.

### 4.1 Test set (frozen, versioned in `benchmark/`)

| Subset | Size | Purpose |
|--------|------|---------|
| `core` | ~2,000 | 10 intents × 200, held-out speakers, clean |
| `oov` | 500 | Out-of-vocabulary / non-command speech (rejection) |
| `noise` | ~500 | Core subset re-rendered at SNR 20/10/5 dB |
| `farfield` | ~300 | RIR-augmented (simulated distance) |

### 4.2 Metrics

**Accuracy:**
- Intent accuracy (top-1)
- Slot F1 (micro + macro) and slot exact-match
- **Command exact match** (intent + all slots correct) — the headline "task success" metric
- WER on the constrained transcript (for ASR-based VCMs)
- Rejection: OOV accuracy, false-accept rate at a fixed confidence threshold

**Robustness:**
- Per-SNR breakdown (20/10/5 dB), far-field vs near-field, per accent

**Efficiency (measured on the target RPi):**
- RTF, p50/p95 end-to-end latency, model size (MB), peak RSS (MB), CPU %, (optional) energy

### 4.3 Harness

- `benchmark/evaluate.py`: takes a model (ONNX) + the frozen test set → emits a JSON
  report; one command, no configuration per team.
- `benchmark/leaderboard.md`: one row per submitted VCM (all metrics above).
- Baselines to seed the leaderboard:
  1. Small open ASR (Vosk small / Whisper-tiny int8) + rule-based parser
  2. SpeechCommands-style keyword spotter + rules
  3. Best VCM from this ME

---

## 5. Validation of our VCM (individual work — item 4)

1. Run `benchmark/evaluate.py` on the frozen set; report all §4.2 metrics.
2. **Ablations:** model size (1M/2M/5M params), int8 vs fp32, data scale (10k/50k/100k),
   augmentation on/off, wake word on/off.
3. **Error analysis:** intent confusion matrix, per-slot error breakdown (which slot
   types fail, at which SNR, which accents).
4. **Real-data check:** evaluate on the held-out real recordings (domain gap).
5. **Efficiency report:** measured RTF/latency/RAM/CPU on the actual RPi4 and RPi5
   (not simulated), with the §2.3 acceptance table.
6. Write-up in `reports/validation.md` with the leaderboard row.

---

## 6. Real-world demo (individual work — item 5; devices may be shared)

**Hardware (BOM):**
- Raspberry Pi 4 (4 GB) or Pi 5 (8 GB), 64-bit Raspberry Pi OS
- Microphone: ReSpeaker 2-Mic (or 4-Mic) HAT, or a USB conference mic
- Output: relay/LED for "lights", small display or web dashboard for state
- Power supply, case

**Software:**
- Python service: VAD → wake word → VCM (ONNX int8) → parser → command JSON
- Mock smart-home backend (local MQTT broker or Flask HTTP): `lights`, `timer`,
  `thermostat`, `music` endpoints that act on the JSON (LED toggles, timer beeps,
  thermostat readout)
- Web dashboard: live transcript, intent, slots, latency, CPU/RAM gauges
- Demo script: 10 commands (one per intent) performed live, plus a rejection demo
  (say something out-of-vocabulary → device does nothing)
- Deliverable: recorded video + `demo/README.md` with build steps

---

## 7. Milestones (6 weeks; adjust to the ME calendar)

| Week | Collective | Individual |
|------|-----------|------------|
| 1 | Finalize taxonomy + template grammars; repo scaffold; TTS pipeline prototype | Choose architecture (A vs B); baseline RNN-T config |
| 2 | Generate dataset v1 (all 10 intents) + augmentation; dataset card | — |
| 3 | **Freeze benchmark set**; build `evaluate.py`; run baselines | Train VCM v1 on dataset v1 |
| 4 | Leaderboard v1; review metrics | First benchmark numbers; error analysis v1 |
| 5 | — | ONNX export + int8; deploy to RPi; latency/RAM profiling; iterate |
| 6 | Final leaderboard | Demo build + final validation report + write-up |

**Definition of done (ME):**
1. Dataset + dataset card merged (collective)
2. VCM trained, ≤ 10 MB int8, meets §2.3 on RPi4 (individual)
3. Benchmark set + harness + leaderboard merged (collective)
4. Validation report with all metrics + ablations (individual)
5. Live demo on RPi4/5, zero network at runtime (individual)

---

## 8. Risks & mitigations

| Risk | Mitigation |
|------|------------|
| TTS domain gap (synthetic ≠ human speech) | Multi-engine TTS, augmentation, real-data validation subset; keep real data out of training |
| Slot errors on numbers/times | Balanced slot coverage in training; constrained decoding; digit-focused augmentation |
| RPi4 CPU budget missed | int8 quantization, smaller model, wake-word gating, frame batching; Option B fallback |
| Benchmark gaming | Frozen, versioned test set; harness is the only scoring path |
| Mic quality / room noise in demo | ReSpeaker array + RIR-augmented training; demo in a controlled room |
| Shared-device scheduling (RPi) | Book device slots in a shared calendar; demo is self-contained (one boot, one script) |

---

## 9. Repo layout

```
Voice-Controlled-Smart-Device/
├── PLAN.md                 # this document
├── README.md
├── data/
│   ├── templates/          # per-intent template grammars
│   ├── generate/           # TTS + augmentation scripts (seeded, reproducible)
│   ├── manifests/          # train/val/test JSONL (audio lives on shared storage)
│   └── DATASET_CARD.md
├── benchmark/
│   ├── testset/            # frozen core/oov/noise/farfield manifests
│   ├── evaluate.py
│   └── leaderboard.md
├── model/
│   ├── train.py            # RNN-T / TCN training
│   ├── configs/
│   └── checkpoints/
├── deploy/
│   ├── export_onnx.py
│   ├── quantize.py
│   └── rpi_service/        # VAD → wake word → VCM → parser → backend
├── demo/
│   ├── mock_home/          # local MQTT/Flask device backend
│   ├── dashboard/
│   └── README.md
└── reports/
    └── validation.md
```

**Hard rules (apply to all code):** no cloud API calls at runtime; no LLM; all inference
on the RPi; dataset-generation TTS is offline and never part of the runtime path.
