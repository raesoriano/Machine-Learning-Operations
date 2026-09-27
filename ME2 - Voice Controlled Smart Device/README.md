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

Every positive audio row carries a **canonical target phrase** — the short
in-vocab phrase the model is trained to emit for that command. The benchmark
compares (intent, slots) after parsing, so paraphrase variety in the audio
still trains one clean target per command.

## Repository layout

```
ME2 - Voice Controlled Smart Device/
├── README.md                     # this file
└── data/
    ├── .gitignore                # audio ignored; manifests+metadata tracked
    ├── manifests/                # P0 deliverables (tracked)
    │   ├── positive_negative_manifest.csv   # 82,778 rows (49,786 pos / 32,992 neg)
    │   ├── frozen_test_v1.jsonl             # frozen benchmark test set (11,492 rows)
    │   └── summary.json
    ├── scripts/
    │   └── build_manifest.py     # regenerates the above from raw sources
    ├── optionb/                  # Mark dataset (31 class dirs + FLAGGED/, audio not tracked)
    │   ├── manifest.csv          # 21,224 rows: 17,924 optionb + 3,300 external curated
    │   ├── labels.json, slots.json, dataset_design.txt, OPTIONB_DATA_SUMMARY.md
    ├── external/                 # external datasets (audio not tracked)
    │   ├── slurp/                #   real 72,395 + synth 10,272 FLAC + parquet + metadata
    │   ├── fluent_speech_commands/  # full 30,043 WAV + curated 900
    │   ├── google_speech_commands_v2/  # 105,835 WAV (100k cmds + 5,835 bg noise)
    │   ├── librispeech/          #   1,500 curated FLAC (OOD speech)
    │   ├── librispeech_test_clean/  # 2,620 FLAC (1,120 held out for frozen OOD test)
    │   └── DATASETS.md           # sources, licenses, re-download
    └── metadata/
        ├── standalone/           # semantic mapping of all external datasets
        │   ├── mapping_rules.json    # the mapping policy (per dataset, per label)
        │   ├── dataset_mapping.csv   # 277,528 row-level decisions (git-ignored, 32 MB)
        │   ├── processed_metadata.csv# 67,189 included rows (git-ignored, 14 MB)
        │   └── plan_stats.json, dataset_inspection.*, standalone_reports/
        ├── moonshine/            # Moonshine baseline utterance list
        └── additional_data_requirements.md   # D1–D5 deficiency analysis
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
- [ ] P2 train tiny CTC on the manifest (HPC A100), ONNX int8 < 2 MB
- [ ] P3 benchmark: per-intent/command accuracy, slot F1, OOD rejection rate,
      latency/RTF on x86 + RPi (harness in the VCM repo)
- [ ] P4 validate on frozen test v1
- [ ] P5 RPi4/5 demo: VAD → VCM → parser → GPIO (LED/relay/buzzer/DHT11)
- [ ] P6 move VCM code into ME2 (single project home)

Model code currently lives in the private `Voice-Controlled-Smart-Device` repo
(same shared storage); it moves here in P6.
