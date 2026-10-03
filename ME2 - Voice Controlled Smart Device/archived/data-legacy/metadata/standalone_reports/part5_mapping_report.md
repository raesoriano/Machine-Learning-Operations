# Part 5 — Dataset-to-Canonical Mapping Report

**Date:** 2026-09-21
**Ontology:** `configs/commands.yaml` v2 (32 classes — authoritative)
**Scope:** semantic mapping of every utterance in the three downloaded datasets to the canonical 32-class ontology. No audio was copied or modified; raw archives were only streamed/read.

## 1. Method

- **Not blind word matching.** Each source label/utterance was judged by its actual meaning:
  - **Fluent Speech Commands** — all 248 unique transcripts were reviewed; mapping is keyed on the dataset's own `(action, object)` slot annotations with explicit phrase overrides (e.g. "Turn off the music" → STOP, not PAUSE).
  - **SLURP** — mapping is keyed on the dataset's intent annotation (its semantic label); slot values are extracted from the dataset's own entity spans (`time`, `color_type`) and cross-checked against the sentence text; sentence-level cues resolve ambiguous intents (e.g. `datetime_query` → TIME only when the sentence asks for the time, not the date).
  - **Google Speech Commands v2** — single-word labels with no transcripts; only words whose meaning is unambiguous in the ontology are mapped.
- **Ambiguity policy — nothing forced:**
  - `UNKNOWN` = in-domain command, but no canonical class fits confidently (directional brightness/temperature commands, mute, bare control words `on/off/up/down`, alarms with non-canonical times, reminders with non-canonical values).
  - `EXCLUDE` = out of the project's command domain (change language, bring items, email, taxi, QA, greetings, calendar events, shopping lists, ...).
- **Speaker IDs** are preserved as metadata only (`speaker_id` column) — never a class. SLURP synthetic recordings have no speaker in `metadata.json` and are left blank.

## 2. Outputs

| File | Description |
|---|---|
| `data/metadata/dataset_mapping.csv` | 277,528 rows — one per audio utterance. Columns: `source_dataset, source_label, source_audio, transcript, canonical_label, slot_type, slot_value, mapping_confidence, include, speaker_id, source_split` |
| `data/metadata/dataset_mapping_stats.json` | statistics + imbalance analysis |
| `data/metadata/mapping_rules.json` | the semantic mapping rules and rationale per dataset |
| `scripts/map_datasets.py` | re-runnable mapping script (read-only on raw data) |

## 3. Headline statistics

| Metric | Count |
|---|---|
| Total utterances mapped | **277,528** |
| Included (canonical class or UNKNOWN) | **67,189** |
| Excluded (out of domain) | **210,339** |
| UNKNOWN (in-domain, no confident class) | **31,631** |

### Per source dataset

| Dataset | Total | Included | Excluded |
|---|---:|---:|---:|
| Fluent Speech Commands | 30,043 | 24,223 | 5,820 |
| SLURP (real + synthetic) | 141,649 | 23,864 | 117,785 |
| Google Speech Commands v2 | 105,836 | 19,102 | 86,734 |

### Mapped utterances per canonical class

| Class | Included | Speakers | By dataset |
|---|---:|---:|---|
| PLAY_MUSIC | 6,572 | 228 | fluent 912, slurp 5,660 |
| WEATHER | 4,855 | 140 | slurp 4,855 |
| LIGHT_OFF | 4,583 | 210 | fluent 3,236, slurp 1,347 |
| STOP | 4,392 | 92 | fluent 510, gsc 3,872, slurp 10 |
| LIGHT_ON | 4,310 | 142 | fluent 4,135, slurp 175 |
| VOLUME_UP | 4,167 | 198 | fluent 3,010, slurp 1,157 |
| VOLUME_DOWN | 3,366 | 184 | fluent 2,731, slurp 635 |
| TIME | 1,849 | 121 | slurp 1,849 |
| PAUSE | 315 | 87 | fluent 315 |
| LIST_REMINDERS | 273 | 65 | slurp 273 |
| ALARM_6_AM | 216 | 52 | slurp 216 |
| COLOR_BLUE | 225 | 50 | slurp 225 |
| ALARM_8_AM | 157 | 40 | slurp 157 |
| COLOR_RED | 148 | 48 | slurp 148 |
| COLOR_GREEN | 73 | 21 | slurp 73 |
| NEXT | 28 | 10 | slurp 28 |
| ALARM_9_PM | 21 | 6 | slurp 21 |
| CALL | 8 | 0 (synthetic) | slurp 8 |
| **UNKNOWN** | **31,631** | 230 | fluent 9,374, gsc 15,230, slurp 7,027 |
| MESSAGE | 0 | — | — |
| TIMER_10_SEC / TIMER_30_SEC / TIMER_1_MIN | 0 | — | — |
| TEMPERATURE_18 / TEMPERATURE_22 / TEMPERATURE_26 | 0 | — | — |
| BRIGHTNESS_20 / BRIGHTNESS_60 / BRIGHTNESS_100 | 0 | — | — |
| REMINDER_DRINK_WATER / REMINDER_STUDY / REMINDER_EXERCISE | 0 | — | — |

## 4. Severe class imbalance — identified

1. **13 of 32 classes have ZERO coverage** in all three datasets:
   `MESSAGE`, all 3 `TIMER_*`, all 3 `TEMPERATURE_*`, all 3 `BRIGHTNESS_*`, all 3 `REMINDER_*`.
   None of the datasets contains utterances with these exact slot values (e.g. no "set a timer for 30 seconds", no "set the temperature to 22 degrees"). **These classes must be synthesized (TTS) or recorded before training.**
2. **Extreme ratio among covered classes:** max/min = **3,954×** (UNKNOWN 31,631 vs CALL 8). Excluding UNKNOWN, PLAY_MUSIC (6,572) vs CALL (8) is still **821×**.
3. **Thin classes:** CALL (8), NEXT (28), ALARM_9_PM (21), COLOR_GREEN (73), PAUSE (315), LIST_REMINDERS (273) — all under ~300 utterances.
4. **UNKNOWN is the largest "class" (31,631)** — a strong rejection signal, but it is dominated by GSC bare words (15,230) and directional brightness/temperature commands (16,464).

**Consequence for training:** per-class balancing (or dropping the weakest classes) and synthetic data generation for the 13 zero-coverage classes are required before Part 6+.

## 5. Notable mapping decisions (examples)

| Utterance (source) | Mapped to | Why |
|---|---|---|
| "Turn the lights on" (Fluent) | LIGHT_ON | direct light-on command |
| "Switch on the lamp" (Fluent) | LIGHT_ON | lamp = light |
| "Turn up the temperature" (Fluent) | UNKNOWN | directional, no canonical 18/22/26 value — not forced |
| "Turn off the music" (Fluent) | STOP | stopping playback, not pausing |
| "Pause the music" (Fluent) | PAUSE | pause semantics |
| "Change language" (Fluent) | EXCLUDE | device setting, out of domain |
| "Bring me my shoes" (Fluent) | EXCLUDE | fetch-item, out of domain |
| "what time is it in florida" (SLURP) | TIME | asks for the time |
| "what date is today" (SLURP) | EXCLUDE | date query, not time |
| "do i need an umbrella today" (SLURP) | WEATHER | indirect weather question |
| "it's too dark here" (SLURP) | LIGHT_ON | darkness cue → turn light on |
| "time to sleep olly" (SLURP) | LIGHT_OFF | sleep cue → turn lights off |
| "dim the lights in the kitchen" (SLURP) | UNKNOWN | directional brightness, no level value |
| "set the colour of light to red" (SLURP) | COLOR_RED | exact color entity |
| "change the light color to yellow" (SLURP) | EXCLUDE | non-canonical color |
| "wake me up at eight o'clock" (SLURP) | ALARM_8_AM | time entity = 8 AM |
| "set an alarm for seven am" (SLURP) | UNKNOWN | alarm with non-canonical time |
| "remind me to pay the bill on weekend" (SLURP) | UNKNOWN | reminder with non-canonical value |
| "call my wife" (SLURP) | CALL | direct call request |
| "call the nearest pizza place" (SLURP) | EXCLUDE | takeaway order, not a person call |
| "stop" (GSC) | STOP | unambiguous |
| "off" (GSC) | UNKNOWN | bare control word — target ambiguous |
| "on" / "up" / "down" (GSC) | UNKNOWN | bare control words |
| "yes" / "no" / "left" / "cat" / "seven" (GSC) | EXCLUDE | not commands in this ontology |

## 6. Caveats

- GSC has no transcripts; its mapping rests on the word's meaning alone (hence the low 0.6 confidence for bare control words).
- SLURP `weather_query` includes indirect questions ("do i need a jacket?") — mapped at 0.85 confidence; non-weather sentences in that intent were excluded.
- CALL/MESSAGE coverage is essentially nil (8 / 0). MESSAGE has no confident source utterances at all.
- `mapping_confidence` encodes how defensible each rule is (0.6–0.95); rows below 0.7 should be reviewed before training.
