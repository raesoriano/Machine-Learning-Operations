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

## Current best model — `pi test v5/` (wake-word-gated PocketSphinx ensemble)

The deployed model is the **PocketSphinx ensemble** from the vcm-v2 work, now
gated by an **openWakeWord wake word** and closed the loop with **spoken TTS
responses**. It is fully self-contained in `pi test v5/` and runs comfortably
on a Pi 4/5 (no Whisper / ONNX / torch).

**Flow (a three-stage state machine):**

```
STANDBY  "hey rhasspy" detector (openWakeWord, always on, ~3 ms/frame)
   │  score ≥ 0.5 over a rolling 0.4 s window
   ▼
CUE     play "yes?" (00_yes.wav) — the mic is muted while it plays,
        so the wake word's own tail can never be decoded as a command
   ▼
COMMAND wait up to 1.0 s for speech to START (webrtcvad; 0.6 s of
        silence ends the utterance) → ensemble decode → stage-2
        classify → play the matching response WAV
   ▼
STANDBY 0.5 s cooldown (response tail can't re-trigger), then wait
        for the next wake word
```

The wake word is **mandatory**: if the model cannot load (missing file,
unfetched Git-LFS pointer, failed self-test), `vcm_pi_v5.py` **refuses to
start** with a loud error instead of silently falling back to always-listen.
That fallback was the old behavior that produced *ghost commands* — noise
decoded into commands on its own. With the gate, the grammar-constrained
decoder simply isn't listening until the wake word fires, so noise and a
background call can no longer produce a command.

### Components

| Part | What it is | Where |
|---|---|---|
| Wake word | openWakeWord **"hey rhasspy"** (204 KB ONNX; scores ~0.8–0.9 on the phrase, ~0.002 on real mic noise; threshold 0.5) | `pi test v5/wakeword/hey_rhasspy_v0.1.onnx` |
| VAD | webrtcvad (16 kHz, 0.6 s silence ends the utterance) | in `vcm_pi_v5.py` |
| Acoustic model A | **custom** 1.6 MB LDA AM trained on the ME2 dataset (`-silprob 0.65 -wip 0.65`) | `pi test v5/am/custom/` |
| Acoustic model B | **stock** 6.4 MB `en-us` AM (`-silprob 0.45 -wip 0.60`) | `pi test v5/am/stock/`, `am/stock_enus/` |
| Grammar | the 103-phrase JSGF command grammar (both AMs decode the same grammar) | `pi test v5/vcm_commands_enh3.jsgf` |
| Dictionary | word→phone dictionary for the custom AM | `pi test v5/dict3` |
| Stage-2 classifier | text classifier over word (1,2) + char (2,5) n-grams, 31 commands + REJECT (5.2 MB) | `pi test v5/classifier.pkl` |
| Responses | 19 TTS WAVs (Piper `en_US-lessac-medium`, 16 kHz mono) + the "yes?" cue | `pi test v5/responses/` |

**Fusion:** the two decoders decode the same grammar; the final command is
the one they **agree on**, or — when they disagree — the one with the
**higher stage-2 classifier confidence** (test-set-agnostic).

### Results

| Set | Command | Intent |
|---|:---:|:---:|
| 171-clip held-out set, one new speaker (`data/additional_test_data`) | **95.3%** | **97.1%** |
| 176-clip set (171 + 5 REJECT clips) | 93.8% | 95.5% |

Reports: `archived/vcm-v2/backbone/reports/pocketsphinx_ensemble_cmudict.json`
(171-clip) and `pi test v5/test_v5_report.json` (176-clip, per-folder
breakdown).

### Spoken responses

The 31 fine-grained commands map onto **19** TTS response phrases (several
commands share a phrase); **REJECT / unknown → `19_repeat.wav`** ("can you
repeat that?"). `00_yes.wav` ("yes?") is the post-wake-word cue, not bound to
a command. Full mapping table in `pi test v5/README.md`.

### Install & run (on the Pi)

```bash
cd "ME2 - Voice Controlled Smart Device/pi test v5"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # openwakeword==0.4.0 is pinned (0.5/0.6 need tflite-runtime, no Py3.13/aarch64 wheel)
python vcm_pi_v5.py --wake-check       # self-test: feeds the bundled "hey rhasspy" fixture through the model
python vcm_pi_v5.py                    # live: STANDBY -> "hey rhasspy" -> "yes?" -> command -> speak
```

Useful flags: `--file clip.wav` (classify one file), `--test` (run the
held-out set), `--no-play`, `--wake-threshold 0.6` (stricter gate),
`--command-window 1.5` (wait longer for the command), `--no-wake`
(always-listen, **debugging only**).

The wake model is stored in Git LFS. If the file on disk is an unfetched
pointer (sub-KB), the program **auto-downloads** the real 204 KB model from
GitHub's raw endpoint on startup (needs internet once); `git lfs pull` works
too. `--wake-check` reports the exact failing step (missing file / LFS
pointer / import / load / inference) if it can't pass.

### Latency & decoder warm-up

Ensemble decode is **~90–120 ms** end-to-end per command on a Pi 5 CPU
(eval `asr_ms_p50 ≈ 87 ms`), plus response playback. A freshly created
PocketSphinx decoder needs a few real-speech utterances to lock in, so the
**first** real command is re-decoded `WARMUP_REPS` (4) times and the
converged result is taken — this fixes the first command and warms the
decoder for all that follow. (Warming with the Piper TTS voice was tried and
rejected — a different speaker biases the decoder.)

### Known limitations

* A few confusable pairs still slip through, e.g. **TIME → PLAY_MUSIC**
  (both are short, single-intent phrases). Candidate fixes: grammar/LM
  tuning, or reweighting the stage-2 classifier on the confusion pairs.
* The wake word must be audible over background noise (a loud zoom call can
  mask it) — that is the inherent trade-off of gating on a spoken phrase.

## Repository layout

```
ME2 - Voice Controlled Smart Device/
├── README.md                     # this file
├── pi test v5/                   # *** CURRENT BEST MODEL *** (wake-word-gated PocketSphinx ensemble)
│   ├── vcm_pi_v5.py              #   listener: wake word -> "yes?" -> VAD -> ensemble -> classify -> speak
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
│   ├── test_wake_flow.py         #   wake-word gate test (synthetic 48 kHz stream)
│   ├── test_v5_report.json       #   176-clip held-out eval (per-folder)
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

## Model history

The current best model above won by a wide margin over two earlier attempts.
Both are frozen in `archived/` with their full reports.

### v1 — from-scratch tiny CTC encoder (`archived/model/`, `me2_v5`)

**Pipeline:** `16 kHz audio -> log-mel [T,40] -> TCN encoder -> CTC logits
[T,186] -> greedy CTC collapse -> transcript -> parser -> (intent, slots)`.
A ~1M-param TCN (Conv1d stem + 4 dilated residual blocks) emits a short
transcript from a constrained 185-word vocab; a pure-Python parser maps it to
(intent, slots). 40 log-mel filters, 25 ms window / 10 ms hop, `n_fft=512`.
Training: AdamW (lr 3e-4), precomputed log-mel `.npz`, early stopping on val
CTC. int8 ONNX ≈ 0.48 MB; p50 latency 14.2 ms on CPU.

**Frozen test v1** (11,492 rows: 6,470 in-domain + 5,022 OOD):

| Metric | Pre-Plan-A (v2) | **me2_v5 (final)** |
|--------|:---:|:---:|
| In-domain exact match (intent+slots) | 18.0% | **44.3%** |
| Intent accuracy | 36.4% | **56.4%** |
| Slot F1 | 31.3% | **55.3%** |
| Command accuracy (overall) | — | **57.2%** |
| OOD rejection rate | 49.8% | **73.7%** |
| WER (transcript-level) | 63.6% | 81.6% |

Full per-intent breakdown in `archived/reports/me2_v5_frozen_int8.json`.
**Why it lost:** it underfit even its own in-domain data (47.7% WER on the
ME2 test split) — a ~1M-param encoder can't learn the 31 commands from
~50k rows when most classes have a few hundred examples.

### v2 — pretrained Whisper `base.en` backbone (`archived/vcm-v2/`)

Fine-tuned a pretrained Whisper base.en ASR backbone + the same 31-command
classifier. **85.4% command / 86.0% intent** on the held-out 171-clip
new-speaker set (vs 24.6% for every from-scratch attempt, 81.9% zero-shot).
**Why it was superseded:** 85.4% was good, but the PocketSphinx ensemble
(95.3%) beat it by ~10 points while being ~10× smaller, with no torch/ONNX
dependency on the Pi. The vcm-v2 work is also where the PocketSphinx
ensemble was discovered — its experiment reports live in
`archived/vcm-v2/backbone/reports/`.

### v3 — current: wake-word-gated PocketSphinx ensemble (`pi test v5/`)

95.3% / 97.1% (see above). The wake-word gate (added 2026-09-30) fixed the
last remaining failure mode — ghost commands from a noisy mic — by making the
decoder arm only after "hey rhasspy".

## Status / plan

- [x] P0 dataset: migration to ME2, GSCv2 extraction, 31→10 mapping from
      metadata, positive/negative manifest, frozen test set, git tracking
- [x] P6 move VCM code into ME2 (single project home) — this repo is the
      only home for the project; the old `Voice-Controlled-Smart-Device`
      folder is retired
- [x] P2 train tiny CTC on the manifest (HPC A100), ONNX int8 < 2 MB —
      Plan A (transcript targets) + 16 kHz pipeline; final model in
      `archived/model/checkpoints/me2_v5/` (int8 0.48 MB)
- [x] P3 benchmark: per-intent/command accuracy, slot F1, OOD rejection
      rate, latency/RTF (harness in `archived/benchmark/`, numbers in
      `archived/reports/`)
- [x] P4 validate on frozen test v1 (final numbers in `archived/reports/`)
- [x] P7 (2026-09-28) VCM v2 backbone: the from-scratch CTC underfits even
      its own in-domain data (47.7% WER on the ME2 test split), so the
      standalone `VCM-v2` repo explored a **pretrained Whisper base.en** ASR
      backbone + the same 31-command classifier. Best result: **85.4% command
      / 86.0% intent** on the held-out 171-clip new-speaker set. Now in
      `archived/vcm-v2/`.
- [x] P8 (2026-09-29/30) **current best model**: PocketSphinx ensemble
      (custom + stock AM, agree+clf fusion) → **95.3% command / 97.1% intent**
      on the 171-clip held-out set; spoken TTS responses (31 commands → 19
      WAVs); openWakeWord "hey rhasspy" gate (mandatory, self-healing model
      load, `--wake-check`); repo restructured around `pi test v5/`.
- [ ] P5 RPi4/5 demo: VAD → VCM → parser → GPIO (LED/relay/buzzer/DHT11)
      (RPi service + simulator in `archived/deploy/`)
- [ ] P9 accuracy pass on the remaining confusion pairs (e.g. TIME →
      PLAY_MUSIC): grammar/LM tuning or stage-2 classifier reweighting
