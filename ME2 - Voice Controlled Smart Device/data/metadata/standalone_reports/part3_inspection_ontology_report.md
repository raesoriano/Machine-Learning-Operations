# Part 3 + 4 — Dataset Inspection & Canonical Command Ontology

**Date:** 2026-09-21
**Scope:** (A) Read-only inspection of the three downloaded raw datasets; (B) definition of the project's authoritative 32-class command ontology.
**Policy:** Raw archives were **streamed, not extracted or modified**. No model was trained.

---

## A. Dataset inspection

Machine-readable results: `data/metadata/dataset_inspection.json` and `data/metadata/dataset_inspection.csv` (one row per dataset, the 15 requested attributes as columns).
Inspection script: `scripts/inspect_datasets.py` (re-runnable; single sequential pass per gzip tar).

### A.1 Summary table

| Attribute | Google Speech Commands v2 | Fluent Speech Commands | SLURP |
|---|---|---|---|
| **Location** | `data/raw/google_speech_commands_v2/` | `data/raw/fluent_speech_commands/` | `data/raw/slurp/` (audio/ + repo/) |
| **Archive** | `speech_commands_v0.02.tar.gz` | `fluentai.zip` | `slurp_real.tar.gz`, `slurp_synth.tar.gz`, `repo/` (git) |
| **Audio format** | WAV (PCM 16-bit) | WAV (PCM 16-bit) | FLAC |
| **Sample rate** | 16 kHz | 16 kHz | 16 kHz |
| **Channels** | 1 (mono) | 1 (mono) | 1 (mono) |
| **Audio files** | 105,835 WAV | 30,043 WAV | 72,395 real FLAC + 69,257 synth FLAC |
| **Duration (p50 / p90 / max)** | 1.00 / 1.00 / 1.00 s (≈1 s clips) | 2.22 / 3.07 / 13.23 s | real 2.56 / 4.48 / 11.33 s; synth 2.17 / 3.36 / 7.31 s |
| **Transcript** | No (label = directory name) | Yes (`transcription` column) | Yes (`sentence` / `sentence_annotation`) |
| **Intent/label** | 35 command words + `_background_noise_` | 31 intents (action×object×location) | 91 intents |
| **Slots** | None | Yes: action, object, location | Yes: typed entity spans (55 entity types) |
| **Speaker IDs** | None (anonymized) | Yes: 97 speakers + demographics | Yes: 174 speakers (`usrid`) |
| **Splits** | **Official**: validation_list (9,981) + testing_list (11,005); 84,849 train | train 23,132 / valid 3,118 / test 3,793 | train / devel / test (+ train_synthetic) |
| **License** | CC-BY-4.0 | Fluent Speech Commands Public License (CC-BY-4.0) | CC-BY-4.0 |
| **Metadata format** | dir/filename + split lists | CSV | JSONL + JSON |
| **Language** | English | English | English |
| **Recorded by** | Human | Human (crowdsourced US/CA) | **Both** (real + synthetic TTS) |

All three datasets are **16 kHz mono English** — directly compatible with each other and with a Pi 5 microphone front-end.

### A.2 Google Speech Commands v2

- **Structure:** `./<class>/<hash>_nohash_<n>.wav` — one directory per command word, flat WAVs.
- **Counts:** 105,835 WAVs across 35 command words + `_background_noise_` (7 non-speech noise files for augmentation). Total archive entries 105,878 (the extra 43 are 36 class dirs, `README.md`, `_background_noise_/README.md`, `LICENSE`, `validation_list.txt`, `testing_list.txt`, `.DS_Store`, `.`).
- **Classes (35 words):** backward, bed, bird, cat, dog, down, eight, five, follow, forward, four, go, happy, house, learn, left, marvin, nine, no, off, on, one, right, seven, sheila, six, stop, three, tree, two, up, visual, wow, yes, zero.
- **Official splits ship in the archive:** `validation_list.txt` (9,981 files) and `testing_list.txt` (11,005 files); the remaining 84,849 files form the train set. Neither list contains `_background_noise_`.
- **No speaker IDs** (anonymized `nohash` filenames) and no free-text transcript — the directory name is the label.
- **License:** CC-BY-4.0 (full text in `LICENSE` inside the archive).

### A.3 Fluent Speech Commands

- **Structure:** `fluent_speech_commands_dataset/wavs/speakers/<speakerId>/<uuid>.wav` + `data/*.csv`.
- **Counts:** 30,043 WAVs from **97 speakers** (readme states 97; `speaker_demographics.csv` has 101 rows including a header-adjacent/extra row). 248 phrases → **31 unique intents** = action × object × location.
- **Splits:** train 23,132 / valid 3,118 / test 3,793 (rows in `train/valid/test_data.csv`).
- **Slots (per row):** `action` ∈ {change language, activate, deactivate, increase, decrease, bring}; `object` ∈ {none, music, lights, volume, heat, lamp, newspaper, juice, socks, shoes, Chinese, Korean, English, German}; `location` ∈ {none, kitchen, bedroom, washroom}.
- **Speaker metadata preserved:** `speakerId` per row + `speaker_demographics.csv` (self-reported fluency, first language, work/school language, gender, ageRange).
- **Durations:** exact, from `train_data_seq2seq.csv` (train split only). `synthetic_data.csv` is empty (header only) — all audio is human-recorded.
- **License:** Fluent Speech Commands Public License (PDF in archive); CC-BY-4.0 per Zenodo record 11106540.

### A.4 SLURP

- **Structure:** real audio = flat `audio-<id>[-headset].flac` in `slurp_real.tar.gz`; synthetic audio in `slurp_synth.tar.gz`; annotations in `repo/dataset/slurp/*.jsonl` + `metadata.json`.
- **Counts:** 72,395 real + 69,257 synthetic FLAC files. Sentences: train 11,514 / devel 2,033 / test 2,974 / train_synthetic 19,711. Recordings per sentence: train 50,628 / devel 8,690 / test 13,078 / train_synthetic 69,253.
- **Labels:** **91 intents** (e.g. `calendar_set`, `play_music`, `weather_query`, `datetime_query`, `alarm_set`, `iot_hue_lighton/off/up/dim/change`, `audio_volume_up/down/mute`, `lists_query/createoradd/remove`, …).
- **Transcript:** `sentence` / `sentence_annotation` fields.
- **Slots:** `entities` field with typed spans — **55 entity types** (date, time, person, place_name, event_name, media_type, color_type, device_type, player_setting, …).
- **Speaker metadata preserved:** `metadata.json` maps each recording file → `usrid` (anonymized, e.g. `FO-488`), **174 unique speakers**.
- **Both real and synthetic** audio provided in separate archives (synthetic = TTS for augmentation).
- **License:** CC-BY-4.0 (`audio/LICENSE.txt`, `repo/LICENSE.txt`).

### A.5 Speaker-identity note (per task constraint)

Speaker IDs are **preserved as metadata only** (Fluent `speakerId` + demographics; SLURP `usrid`). They are **not** classification targets. They will be used later solely to build **speaker-disjoint** train/validation/test splits so the model is not evaluated on the same voice it was trained on. GSC has no speaker IDs, so its speaker-disjoint split is not applicable (its official split lists are used instead).

### A.6 Relevance to the 32-class ontology

None of the three datasets uses our exact 32 classes. They are **source material** for later mapping/synthesis:
- **GSC** contributes single-word commands (on/off, yes/no, up/down, left/right, stop, go, …) → candidates for fixed classes and for `_background`-style rejection.
- **Fluent** contributes slotted, natural-language commands (activate/deactivate lights, increase/decrease volume/heat, …) → the closest match to our fixed + slotted command style.
- **SLURP** contributes the richest intent + entity coverage (91 intents, 55 entity types) and both real and synthetic audio → the main source for slotted-command phrasings (timers, alarms, temperature, brightness, color, reminders).

Mapping dataset labels → the 32 canonical classes is a later step (canonical dataset creation); it is **not** performed here.

---

## B. Canonical command ontology

Authoritative file: **`configs/commands.yaml`** (version 2, supersedes the v1 11-word label set from Part 1).

### B.1 Composition

| Group | Count | Classes |
|---|---|---|
| Fixed commands | 13 | PLAY_MUSIC, WEATHER, TIME, LIGHT_ON, LIGHT_OFF, PAUSE, STOP, NEXT, VOLUME_UP, VOLUME_DOWN, CALL, MESSAGE, LIST_REMINDERS |
| Slotted — TIMER | 3 | TIMER_10_SEC, TIMER_30_SEC, TIMER_1_MIN |
| Slotted — ALARM | 3 | ALARM_6_AM, ALARM_8_AM, ALARM_9_PM |
| Slotted — TEMPERATURE | 3 | TEMPERATURE_18, TEMPERATURE_22, TEMPERATURE_26 |
| Slotted — BRIGHTNESS | 3 | BRIGHTNESS_20, BRIGHTNESS_60, BRIGHTNESS_100 |
| Slotted — COLOR | 3 | COLOR_RED, COLOR_BLUE, COLOR_GREEN |
| Slotted — CREATE_REMINDER | 3 | REMINDER_DRINK_WATER, REMINDER_STUDY, REMINDER_EXERCISE |
| UNKNOWN | 1 | UNKNOWN |
| **Total** | **32** | 13 fixed + 18 slot-value + 1 UNKNOWN |

- **19 command categories** = 13 fixed + 6 slotted intents (TIMER, ALARM, TEMPERATURE, BRIGHTNESS, COLOR, CREATE_REMINDER).
- Each slotted intent is **expanded into 3 separate classes** (one per slot value) for the initial speech classifier → 18 slot-value classes.
- **UNKNOWN** is the rejection class (ambient speech / noise / out-of-ontology commands).

### B.2 Schema

Every class in `configs/commands.yaml` carries:
- `id` (class ID, e.g. `TIMER_10_SEC`)
- `name` (human-readable, e.g. "Set 10 second timer")
- `parent_intent` (one of the 19 categories, or `null` for UNKNOWN)
- `slot_type` (`duration` / `time` / `temperature` / `level` / `color` / `reminder`, or `null`)
- `slot_value` (e.g. `"10 seconds"`, `"6:00 AM"`, `"red"`, or `null`)
- `examples` (3 representative phrases; empty for UNKNOWN)

### B.3 Authoritative label order

`label_order` in the YAML is the single source of truth for the **class index** used in training and inference (index = position). It lists all 32 IDs in the order: 13 fixed → 18 slot-value (grouped by intent) → UNKNOWN.

### B.4 Validation

`configs/commands.yaml` was validated programmatically:
- `num_classes` field == 32, `classes` list length == 32, `label_order` length == 32
- No duplicate class IDs; `label_order` exactly matches the `classes` order
- 13 fixed + 18 slotted + 1 UNKNOWN; 6 distinct slotted intents
- Every class has all required fields (`id`, `name`, `parent_intent`, `slot_type`, `slot_value`, `examples`)

No additional command categories were invented. Speaker identity is **not** a class.

---

## Files changed / created (this part)

| Path | Action |
|---|---|
| `configs/commands.yaml` | **Rewritten** — v2, authoritative 32-class ontology |
| `data/metadata/dataset_inspection.json` | **New** — machine-readable inspection (all 15 attributes × 3 datasets) |
| `data/metadata/dataset_inspection.csv` | **New** — one row per dataset, 15 attribute columns |
| `scripts/inspect_datasets.py` | **New** — re-runnable read-only inspection script |
| `.gitignore` | **Updated** — whitelist the two new `data/metadata/` files |
| `reports/part3_inspection_ontology_report.md` | **New** — this report |

**Not done (per task):** no datasets downloaded, no raw data modified/extracted, no model trained, no dataset→ontology mapping performed (that is a later canonical-dataset step).
