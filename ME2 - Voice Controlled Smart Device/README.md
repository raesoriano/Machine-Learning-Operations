# ME2 — Voice-Controlled Smart Device

Build a **tiny Voice Command Model (VCM)** that understands the 10 most common
command intents people give smart devices, runs **on-device in real time**
(RPi4/5), fully **standalone** (no cloud, no LLM).

## The 10 intents / 31 commands

| # | Intent | Commands (31) |
|---|--------|---------------|
| 1 | `play_music` | PLAY_MUSIC |
| 2 | `ask_question` | WEATHER, TIME |
| 3 | `lights_switch` | LIGHT_ON, LIGHT_OFF |
| 4 | `lights_adjust` | BRIGHTNESS_20/60/100, COLOR_RED/GREEN/BLUE |
| 5 | `set_timer` | TIMER_10s/30s/1m |
| 6 | `set_alarm` | ALARM_6AM/8AM/9PM |
| 7 | `set_temperature` | TEMPERATURE_18/22/26 |
| 8 | `media_control` | PAUSE, STOP, NEXT, VOLUME_UP, VOLUME_DOWN |
| 9 | `reminders_lists` | LIST_REMINDERS, CREATE_REMINDER_DRINK_WATER/EXERCISE/STUDY |
| 10 | `call` | CALL, MESSAGE |

Plus a reject class: **unknown** (anything else must be rejected, not guessed).

## Approach (decided 2026-09-27)

**Option A — tiny CTC encoder → greedy decode → deterministic parser → (intent, slots).**
A ~1M-param CTC encoder emits a short transcript from a **constrained 185-word
vocab**; a pure-Python parser maps the transcript to (intent, slots). Slot
values ("6 AM", "22 degrees") come from the transcript, which a flat
classifier cannot do. (Option B = 32-class CRNN classifier, kept as the
comparison baseline; Option C = hybrid, only if needed.)

**Training target (Plan A):** the CTC target is the row's **transcript** —
the words the audio actually says — not the canonical phrase. Training on the
canonical target (the pre-Plan-A recipe) capped train exact-match at ~17%
because the audio->text mapping was inconsistent ("Alarm 6 AM" -> "set an
alarm for six am"). With transcript targets the mapping is consistent, the
parser recovers (intent, slots), and the benchmark compares (intent, slots)
after parsing, so paraphrase variety in the audio is fine.

## Model design

**Pipeline:** `16 kHz audio -> log-mel [T,40] -> TCN encoder -> CTC logits
[T,186] -> greedy CTC collapse -> transcript -> parser -> (intent, slots)`.

* **Audio standard: 16 kHz everywhere.** Every dataset in the manifest
  (OptionB, Fluent Speech Commands, GSCv2, SLURP, LibriSpeech) is 16 kHz mono
  PCM_16; the RPi mic captures at 16 kHz; the VAD runs at 16 kHz. The mel
  filters are capped at 8 kHz, so 16 kHz Nyquist is exactly sufficient.
  `vcm/features.py` is the **single source of truth** for features — training,
  the ONNX runtime, and the RPi service all import the same `log_mel()`, so
  train/serve features can never drift. (The loader resamples only as a
  defensive no-op; nothing in the real pipeline resamples.)
* **Features:** 25 ms Hann window, 10 ms hop, 40 log-mel filters
  (50 Hz–8 kHz), `log1p` scaling, `n_fft=512`.
* **Encoder (`model/model_def.py`):** Conv1d stem (40→128, k=5) + 4 residual
  GRU-free TCN blocks (dilated 1/2/4/8, k=3) + Linear(128→186). ~1M params;
  float32 ≈ 1.2 MB, int8 ≈ 0.36 MB — well under the 10 MB RPi budget.
* **CTC:** blank=0, words 1..185. Loss = `nn.functional.ctc_loss`
  (log-softmax, zero_infinity). Greedy decode = argmax per frame, remove
  blanks, merge repeats (`model/decode.py`); a small beam search is available
  for hard cases. Because the output head only has |VOCAB|+1 units, the model
  can *only* emit command words — the constrained decoding that makes this a
  "pure" VCM.
* **Training recipe (v4/v5):** AdamW (lr 3e-4, wd 1e-4), batch 64, 300-epoch
  cap with early stopping (patience 24 on val CTC) and ReduceLROnPlateau
  (factor 0.5, patience 6, min 1e-5). Cosine decay was tried first (v2/v3)
  and killed by early stopping while LR was still 68–96% of peak — the
  plateau scheduler is what lets the model finish fitting the data.
  Precomputed log-mel `.npz` features (scripts/build_train_features.py) make
  training I/O-free.
* **Rejection:** not a CTC class — the benchmark scores OOD rows by whether
  the decoded transcript parses to a valid command (false-accept rate).

## Repository layout

```
ME2 - Voice Controlled Smart Device/
├── README.md                     # this file
├── pi test v5/                   # *** CURRENT BEST MODEL *** (PocketSphinx ensemble, runs on Pi)
│   ├── vcm_pi_v5.py              #   listener: wake word -> VAD -> ensemble -> classify -> speak
│   ├── am/custom/                #   custom 1.6 MB LDA acoustic model (+ vcm.lm.bin)
│   ├── am/stock/                 #   stock en-us acoustic model
│   ├── am/stock_enus/            #   cmudict + en-us.lm.bin
│   ├── dict3                     #   word->phone dictionary for the custom AM
│   ├── vcm_commands_enh3.jsgf    #   the 103-phrase command grammar
│   ├── classifier.pkl            #   stage-2 text classifier (31 commands + REJECT)
│   ├── wakeword/                 #   openWakeWord "hey rhasspy" model + self-test fixture
│   ├── responses/                #   19 TTS response WAVs (+ 00_yes.wav cue)
│   ├── vcm/, vcm2/               #   self-contained code (normalization, classifier, ground truth)
│   ├── training/                 #   AM training + eval scripts (build_trained_am.py, eval_*)
│   ├── test_wake_flow.py         #   wake-word gate test
│   └── requirements.txt
├── data/                         # ALL DATA (audio on shared storage, never committed)
│   ├── additional_test_data/     #   held-out test set (171 clips + 5 REJECT, one new speaker)
│   ├── manifests/                #   P0 deliverables (tracked)
│   ├── optionb/                  #   Mark dataset (31 class dirs, audio not tracked)
│   ├── external/                 #   external datasets (audio not tracked)
│   ├── metadata/                 #   semantic mapping of all external datasets
│   ├── features/                 #   precomputed log-mel features (git-ignored, regenerable)
│   └── scripts/, templates/
└── archived/                     # everything that is NOT the current best model
    ├── model/                    #   old CTC VCM (PyTorch) + checkpoints/me2_v5
    ├── benchmark/                #   frozen + synthetic harness, metrics, ONNX/torch backends
    ├── deploy/                   #   ONNX conversion + int8 quantize, RPi service, simulator
    ├── demo/                     #   mock smart-home dashboard
    ├── scripts/                  #   feature/vocab build scripts
    ├── reports/                  #   historical training/eval reports
    ├── tools/                    #   check_grammar.py CI gate
    ├── vcm/                      #   the "golden rule" contract (features, vocab, parser)
    ├── PLAN.md, COLLECTIVE_TASKS.md
    └── vcm-v2/                   #   superseded VCM v2 (Whisper backbone + old pi_test)
        ├── backbone/             #     Whisper fine-tune pipeline + PocketSphinx experiments
        ├── pi_test/              #     old Whisper-ONNX Pi listener
        ├── test_data/            #     (test set now consolidated into data/additional_test_data)
        ├── archive/ctc_v8/       #     frozen from-scratch CTC pipeline
        ├── archive/w2v2_base/    #     frozen abandoned wav2vec2-base CTC attempt
        └── README.md, MIGRATION.md
```

Audio lives on shared storage and is **never committed** (see `data/.gitignore`).
The two big row-level mapping CSVs are regenerable from `mapping_rules.json` +
the raw datasets, so they are also ignored.

## Dataset (P0, done 2026-09-27)

`positive_negative_manifest.csv` — columns:
`id, audio, polarity, intent, command, target, transcript, slots, split, source, speaker, variant`

* **Positive (49,786):** OptionB 17,924 (31 classes × 100 speakers × clean/noisy)
  + external rows whose metadata canonical label is one of the 31 commands:
  FSC 14,849, SLURP 13,141, GSCv2 3,872 (all `stop`).
* **Negative (32,992):** metadata UNKNOWN/EXCLUDE rows (in-domain-ambiguous +
  out-of-domain speech) + GSCv2 `_background_noise_` + LibriSpeech OOD speech
  (1,500 train / 1,120 held out for the frozen test).
* **Splits:** OptionB keeps its speaker-disjoint 80/10/10 speaker split;
  external keeps source splits. **Frozen test v1 = 11,492 rows**
  (6,470 in-domain positives from the 10 held-out OptionB speakers + 5,022 OOD
  rejects: 3,902 external test-split + 1,120 LibriSpeech test-clean never seen
  in training).
* **Excluded:** OptionB FLAGGED wavs (730, known-command audio that failed QC).
* **Known gaps (documented):** BRIGHTNESS_*/TEMPERATURE_*/CREATE_REMINDER_*
  have OptionB audio only (the metadata policy mapped external directional /
  non-canonical-value rows to UNKNOWN); 4,961 SLURP synth rows reference a
  different SLURP synth distribution than the 10,272 synth FLACs on disk
  (all 18,089 SLURP *real* rows resolved).

Regenerate: `python3 data/scripts/build_manifest.py` (from this folder).

## Status / plan

- [x] P0 dataset: migration to ME2, GSCv2 extraction, 31→10 mapping from
      metadata, positive/negative manifest, frozen test set, git tracking
- [x] P6 move VCM code into ME2 (single project home) — this repo is the
      only home for the project; the old `Voice-Controlled-Smart-Device`
      folder is retired
- [x] P2 train tiny CTC on the manifest (HPC A100), ONNX int8 < 2 MB —
      Plan A (transcript targets) + 16 kHz pipeline; final model in
      `model/checkpoints/me2_v5/` (int8 0.48 MB)
- [x] P3 benchmark: per-intent/command accuracy, slot F1, OOD rejection
      rate, latency/RTF (harness in `benchmark/`, numbers in `reports/`)
- [x] P4 validate on frozen test v1 (final numbers in `reports/`)
- [ ] P5 RPi4/5 demo: VAD → VCM → parser → GPIO (LED/relay/buzzer/DHT11)
      (RPi service + simulator in `deploy/`)
- [x] P7 (2026-09-28) VCM v2 backbone: the from-scratch CTC underfits even
      its own in-domain data (47.7% WER on the ME2 test split), so the
      standalone `VCM-v2` repo explored a **pretrained Whisper base.en** ASR
      backbone + the same 31-command classifier. Migrated into `vcm-v2/`
      (this folder). Best result: **85.4% command / 86.0% intent** on the
      held-out 171-clip new-speaker set (vs 24.6% for every from-scratch
      attempt, 81.9% zero-shot). See `vcm-v2/README.md` + `vcm-v2/MIGRATION.md`.

## Results

**Frozen test v1** (11,492 rows: 6,470 in-domain + 5,022 OOD), final `me2_v5`
model (int8 ONNX). Full per-intent breakdown in
`reports/me2_v5_frozen_int8.json`.

| Metric | Pre-Plan-A (v2) | **me2_v5 (final)** |
|--------|:---:|:---:|
| In-domain exact match (intent+slots) | 18.0% | **44.3%** |
| Intent accuracy | 36.4% | **56.4%** |
| Slot F1 | 31.3% | **55.3%** |
| Command accuracy (overall) | — | **57.2%** |
| OOD rejection rate | 49.8% | **73.7%** |
| WER (transcript-level) | 63.6% | 81.6% |

Per-intent exact match (me2_v5): call 86.6%, media_control 67.4%,
lights_adjust 54.7%, set_timer 49.4%, reminders 45.5%, ask_question 40.1%,
set_alarm 35.7%, lights_switch 34.6%, set_temperature 24.4%, play_music 15.0%.

> **Note on the benchmark gold:** the frozen v1 slot labels were generated by
> an early parser revision. The parser is the project's contract ("golden
> rule"), so the gold slots were re-derived from the current parser
> (`gold_source: parser-canonical`; the original is preserved as
> `frozen_test_v1_original.jsonl`). The parser's *intent* agrees with the
> manifest on 100% of in-domain rows and rejects 100% of OOD rows, so a
> perfect model now scores 100% and the numbers above are self-consistent.

### Inference performance (CPU, int8 ONNX, 300-row sample)

`reports/me2_v5_latency.json` — the same CPU-only ONNX Runtime path the RPi
service uses, so these numbers transfer (RPi is slower than the HPC x86 CPU,
but the margins are large):

| Metric | Value | RPi target |
|--------|:---:|:---:|
| Latency p50 | **14.2 ms** | ≤ 500 ms |
| Latency p95 | 53.7 ms | — |
| Latency p99 | 105.5 ms | — |
| RTF (real-time factor) | **0.0068** | ≤ 0.5 |
| Throughput | 52.4 utt/s | — |

Model size: float32 ≈ 1.7 MB, **int8 ≈ 0.48 MB** (budget ≤ 10 MB).

### Known limitations / next steps

The pipeline is complete end-to-end (audio → features → CTC → parser →
intent/slots) and real-time, but in-domain exact match (44%) is the main
optimization target. Levers, in expected order of impact:
1. **Longer/better training** — the v5 run early-stopped at epoch 101 while
   val CTC was still falling (2.01); a tuned patience + LR schedule should
   lift it.
2. **Data augmentation** — the manifest carries clean+noisy variants; training
   on both (currently a single pass) adds robustness.
3. **Per-intent balancing** — set_temperature (24%) and play_music (15%) are
   the weakest; both are data/format-driven (few OptionB-only rows; the
   play_music query slot was a parser bug, now fixed).
