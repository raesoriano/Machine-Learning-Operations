# Part 6 — Canonical Dataset Plan (pre-copy report)

**Date:** 2026-09-22T05:48:30+00:00  
**Inputs:** `data/metadata/dataset_mapping.csv` (Part 5, 67189 included rows) + read-only archive header scan  
**Ontology:** `configs/commands.yaml` v2 (32 classes)  
**Status:** audio NOT yet copied — this report precedes the build phase.

## 1. Totals

| Metric | Value |
|---|---|
| Included rows in mapping | 67,189 |
| Canonical dataset rows | **67,189** |
| Dropped: bad/unknown class | 0 (none) |
| Dropped: header missing | 0 |
| Total audio duration | 134,534 s (37.37 h) |
| Duplicate source recordings | 0 (asserted) |
| Synthetic duplication for imbalance | none (per policy) |

## 2. Recordings per class

| Class | Total | Fluent | GSC | SLURP real | SLURP synth | Real-human | Speakers | Duration (min / p50 / p90 / max s) | Flags |
|---|---|---|---|---|---|---|---|---|---|
| UNKNOWN | 31,631 | 9,374 | 15,230 | 5,552 | 1,475 | 30,156 | 230 | 0.299 / 1.28 / 3.12 / 13.182 | — |
| PLAY_MUSIC | 6,572 | 912 | 0 | 4,036 | 1,624 | 4,948 | 228 | 0.512 / 2.211 / 3.712 / 18.685 | — |
| WEATHER | 4,855 | 0 | 0 | 3,400 | 1,455 | 3,400 | 140 | 0.64 / 2.304 / 3.84 / 13.166 | — |
| LIGHT_OFF | 4,583 | 3,236 | 0 | 1,055 | 292 | 4,291 | 210 | 0.512 / 2.219 / 3.072 / 6.72 | — |
| STOP | 4,392 | 510 | 3,872 | 7 | 3 | 4,389 | 92 | 0.363 / 1.0 / 1.451 / 4.949 | — |
| LIGHT_ON | 4,310 | 4,135 | 0 | 118 | 57 | 4,253 | 142 | 0.64 / 2.219 / 2.987 / 10.217 | — |
| VOLUME_UP | 4,167 | 3,010 | 0 | 1,059 | 98 | 4,069 | 198 | 0.65 / 2.043 / 3.072 / 9.343 | — |
| VOLUME_DOWN | 3,366 | 2,731 | 0 | 587 | 48 | 3,318 | 184 | 0.64 / 2.048 / 2.88 / 13.227 | — |
| TIME | 1,849 | 0 | 0 | 1,407 | 442 | 1,407 | 121 | 0.512 / 2.176 / 3.328 / 8.447 | — |
| PAUSE | 315 | 315 | 0 | 0 | 0 | 315 | 87 | 0.939 / 1.792 / 2.508 / 4.437 | — |
| LIST_REMINDERS | 273 | 0 | 0 | 209 | 64 | 209 | 65 | 0.896 / 2.559 / 4.084 / 6.016 | — |
| COLOR_BLUE | 225 | 0 | 0 | 135 | 90 | 135 | 50 | 0.896 / 2.176 / 3.712 / 6.207 | — |
| ALARM_6_AM | 216 | 0 | 0 | 197 | 19 | 197 | 52 | 1.024 / 2.56 / 3.872 / 5.73 | — |
| ALARM_8_AM | 157 | 0 | 0 | 140 | 17 | 140 | 40 | 1.152 / 2.816 / 4.927 / 7.296 | — |
| COLOR_RED | 148 | 0 | 0 | 114 | 34 | 114 | 48 | 1.088 / 2.237 / 3.924 / 5.224 | — |
| COLOR_GREEN | 73 | 0 | 0 | 39 | 34 | 39 | 21 | 1.0 / 2.176 / 3.252 / 3.904 | — |
| NEXT | 28 | 0 | 0 | 19 | 9 | 19 | 10 | 1.244 / 1.998 / 2.759 / 3.904 | — |
| ALARM_9_PM | 21 | 0 | 0 | 15 | 6 | 15 | 6 | 1.817 / 2.944 / 4.096 / 4.352 | — |
| CALL | 8 | 0 | 0 | 0 | 8 | 0 | 0 | 1.039 / 1.167 / 1.236 / 1.261 | SYNTHETIC_ONLY |
| TEMPERATURE_18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| REMINDER_DRINK_WATER | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| BRIGHTNESS_20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| TEMPERATURE_26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| TIMER_1_MIN | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| BRIGHTNESS_60 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| MESSAGE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| TIMER_30_SEC | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| TIMER_10_SEC | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| REMINDER_STUDY | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| REMINDER_EXERCISE | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| BRIGHTNESS_100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |
| TEMPERATURE_22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | NO_REAL_RECORDINGS |

## 3. Recordings per dataset

- **fluent_speech_commands**: 24,223
- **slurp**: 23,864
- **google_speech_commands_v2**: 19,102

## 4. Speakers per class

Speaker IDs exist for Fluent (97 speakers) and SLURP (usrid); GSC files are
anonymized (`nohash`), so its speaker_id is empty by design.

| Class | Distinct speakers |
|---|---|
| UNKNOWN | 230 |
| PLAY_MUSIC | 228 |
| LIGHT_OFF | 210 |
| VOLUME_UP | 198 |
| VOLUME_DOWN | 184 |
| LIGHT_ON | 142 |
| WEATHER | 140 |
| TIME | 121 |
| STOP | 92 |
| PAUSE | 87 |
| LIST_REMINDERS | 65 |
| ALARM_6_AM | 52 |
| COLOR_BLUE | 50 |
| COLOR_RED | 48 |
| ALARM_8_AM | 40 |
| COLOR_GREEN | 21 |
| NEXT | 10 |
| ALARM_9_PM | 6 |
| CALL | 0 |

## 5. Duration per class

See the Duration column in §2 (min / p50 / p90 / max, seconds).

## 6. Sample-rate / channel / bit-depth distribution

| Property | Distribution |
|---|---|
| sample_rate_hz | {'16000': 67189} |
| num_channels | {'1': 67189} |
| bits_per_sample | {'16': 67189} |

All sources are already 16 kHz mono 16-bit, so the canonical build is a
byte-copy for WAV files and a lossless FLAC→WAV decode for SLURP — no
resampling or quality loss.

## 7. Original split distribution

- fluent_speech_commands:test: 3,069
- fluent_speech_commands:train: 18,562
- fluent_speech_commands:valid: 2,592
- google_speech_commands_v2:testing: 2,040
- google_speech_commands_v2:train: 15,249
- google_speech_commands_v2:validation: 1,813
- slurp:devel: 2,046
- slurp:test: 3,473
- slurp:train: 12,570
- slurp:train_synthetic: 5,775

## 8. Classes flagged as insufficient real human recordings

Policy: flag if 0 real-human recordings, synthetic-only, fewer than 5 distinct speakers, or any recording shorter than 0.25 s.

| Class | Flags | Detail |
|---|---|---|
| CALL | SYNTHETIC_ONLY | real-human=0, speakers=0, slurp_synth=8 |
| MESSAGE | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| TIMER_10_SEC | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| TIMER_30_SEC | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| TIMER_1_MIN | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| TEMPERATURE_18 | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| TEMPERATURE_22 | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| TEMPERATURE_26 | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| BRIGHTNESS_20 | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| BRIGHTNESS_60 | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| BRIGHTNESS_100 | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| REMINDER_DRINK_WATER | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| REMINDER_STUDY | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |
| REMINDER_EXERCISE | NO_REAL_RECORDINGS | real-human=0, speakers=0, slurp_synth=0 |

Classes with **zero** real recordings in all three datasets must be
covered by synthetic speech (ME2 pipeline) or a re-scoped ontology
before training — they are listed above with `NO_REAL_RECORDINGS`.

## 9. Canonical layout

```
data/processed/
├── metadata.csv            # one row per recording (audio_file relative to data/processed/)
├── plan_stats.json         # machine-readable version of this report
└── audio/
    ├── fluent_speech_commands/<uuid>.wav
    ├── google_speech_commands_v2/<word>/<hash>_nohash_<n>.wav
    └── slurp/{real,synth}/audio-<id>.flac -> .wav
```

Canonical audio format: **WAV, 16 kHz, mono, 16-bit PCM**.
