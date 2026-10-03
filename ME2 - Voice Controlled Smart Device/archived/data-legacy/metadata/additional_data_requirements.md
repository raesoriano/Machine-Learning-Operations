# Part 9 — Additional Data Requirements

**Date:** 2026-09-22
**Scope:** Determine whether external speech datasets are *actually* needed for the
OptionB-derived canonical command dataset, based on the Part 1 inspection, Part 3
adaptation, Part 4 validation, and Part 8 baseline evaluation.
**Constraint honored:** No data was downloaded in this step. No dataset is recommended
to increase raw size or to fill a class that already has adequate coverage.

---

## 1. Verdict (short answer)

**Yes — additional data is needed, but only for three specific, evidenced
deficiencies.** It is *not* needed for class coverage or recording count.

| # | Deficiency | Evidence | Needed? |
|---|-----------|----------|---------|
| D1 | **No out-of-command / UNKNOWN negatives** — the 32nd ontology class is empty | 0 OOD recordings in OptionB; Part 8: rejection is *unmeasured* | **Yes (P0)** |
| D2 | **Only 3 fixed slot values per slotted class** — no numerical/time-expression diversity | Within-family confusion: TEMPERATURE_22→18 (27), BRIGHTNESS_20→100 (14), TIMER_10→30 (12) | **Yes (P1)** |
| D3 | **Only 3 phrase templates per class** — limited phrasing/pronunciation diversity | Weak single-word classes: STOP (F1 0.63), LIGHT_ON (0.64), TIME (0.56) | **Yes (P2)** |
| D4 | Limited acoustic conditions (clean + ~30 dB light noise only) | Part 1 §15: no reverb/far-field/babble | Secondary — augmentation first |
| D5 | Limited accent diversity (84 American + 16 Filipino-English) | Part 1 §8 | Secondary |

**Not deficiencies (do NOT add data for these):**
- **Missing command classes:** none. All 31 in-distribution classes are covered (528–600
  recordings each, ratio 1.14×). Only UNKNOWN (the 32nd class) is absent — that is D1.
- **Insufficient recordings:** 528–600 per class is ample for a 244K-param classifier.
  Adding data "just to be bigger" is explicitly out of scope.

---

## 2. Evidence base — what OptionB already provides (strengths)

From Parts 1/3/4, the dataset is strong on the axes that usually limit command models:

- **31 classes, balanced:** 528–600 recordings each (max/min ratio 1.14×).
- **Speaker coverage:** 100 speakers, and **every speaker recorded every class**
  (100/100 speakers per class). Speaker-disjoint splits (train 80 / val 10 / test 10).
- **Two acoustic conditions:** clean + light background noise (~30 dB SNR).
- **Clean metadata:** 11 original columns + `canonical_label`, 0 missing/invalid,
  0 duplicates, 0 missing audio.

So the gaps are **not** about volume or class coverage — they are about
**what the model has never been asked to do** (reject non-commands), **how little
variety exists in the numbers and phrasings** (3 values, 3 templates), and **how
narrow the acoustic/accent envelope is**.

---

## 3. Identified deficiencies (each tied to Part 8 evidence)

### D1 — No out-of-command / UNKNOWN negatives  *(P0)*
- The ontology has **32 classes**; OptionB contains recordings for **31**. The 32nd,
  **UNKNOWN**, has **zero** examples.
- Part 8 states rejection is *unmeasured*: "OptionB contains no out-of-command speech.
  A softmax-confidence threshold sweep … plus **external OOD clips** is the natural
  Part 9."
- **Consequence:** the baseline can classify the 31 in-distribution commands but has
  never learned to say "that is not a command." In real deployment the device hears
  conversation, TV, and noise far more often than commands. Without OOD negatives the
  model will force every utterance into one of the 31 classes.
- **This is a functional gap, not a size gap.** It requires *new, different* speech
  (non-commands), which OptionB cannot supply by construction.

### D2 — Only 3 fixed slot values per slotted class  *(P1)*
- Each slotted class has exactly **3 slot values** (TIMER: 10 s / 30 s / 1 min;
  ALARM: 6 AM / 8 AM / 9 PM; TEMPERATURE: 18 / 22 / 26; BRIGHTNESS: 20 / 60 / 100;
  COLOR: red / blue / green; REMINDER: drink water / study / exercise), and each value
  appears in only **3 phrase templates**.
- The confusion matrix shows the model **cannot robustly discriminate the numbers**:
  - TEMPERATURE_22 → TEMPERATURE_18: **27** errors (recall 0.37, worst class, F1 0.44)
  - BRIGHTNESS_20 → BRIGHTNESS_100: **14** errors (recall 0.62)
  - TIMER_10_SEC → TIMER_30_SEC: **12** errors
- **Consequence:** the classifier has only ever seen three specific numbers per
  slotted intent. It has not learned *number discrimination in general*, so it
  confuses the close values. More diverse phrasings/speakers of the existing values —
  and (if the ontology is later expanded) other numbers — is the fix.
- **Note (design decision, flagged for Part 10):** the *initial* classifier is fixed at
  32 classes with 3 slot values each. Improving discrimination of those 3 values is in
  scope; **expanding the ontology to more slot values** (e.g., "24 degrees", "7:30 AM")
  is a separate decision that changes the class count and is not forced here.

### D3 — Only 3 phrase templates per class  *(P2)*
- Every class is spoken in exactly **3 fixed templates** (e.g., TIMER_10s →
  "Timer 10 seconds" / "Countdown for 10 seconds" / "Start a timer for 10 seconds").
- The weakest classes are the **single-word / short commands**, where 3 templates give
  the least phonetic variety:
  - **STOP** F1 0.63 (recall 0.53) — confused with PLAY_MUSIC (12), PAUSE (10), NEXT (6)
  - **LIGHT_ON** F1 0.64 (recall 0.61) — confused with LIGHT_OFF (8), VOLUME_UP (6)
  - **TIME** F1 0.56 (recall 0.50) — confused with LIST_REMINDERS (9), MESSAGE (9),
    REMINDER_STUDY (7)
  - **PAUSE↔STOP** 12 errors, **LIGHT_ON↔LIGHT_OFF** 21 errors (worst pair)
- **Consequence:** the model is partly overfit to three phrasings per command. Real
  users say "stop it", "halt", "kill the lights", "lights on!", "what's the time".
  More phrasing/pronunciation variety for the weak single-word commands is the fix.

### D4 — Limited acoustic conditions  *(secondary — augment first)*
- Only clean + light noise (~30 dB). No reverb, far-field, babble, or music noise.
- **Preferred fix is augmentation** (add noise/reverb to existing clips) rather than a
  new dataset, because the command *content* is already covered. A real-world
  recording-condition dataset is a secondary option.

### D5 — Limited accent diversity  *(secondary)*
- 100 speakers is a healthy count, but the accent mix is narrow: 84 American
  (LibriSpeech) + 16 Filipino-English (SilencioPH). Broadening accents is a
  secondary robustness goal, not a blocking gap.

---

## 4. Recommended datasets — each mapped to a specific deficiency

Only datasets that address a **clearly identified** deficiency are listed. Each entry
states *exactly* which OptionB deficiency it fixes and *how*.

### 4.1 Common Voice (Mozilla) — **addresses D1** (and secondarily D4, D5)
- **Deficiency fixed:** **D1 — no OOD/UNKNOWN negatives.**
- **How:** Common Voice is *conversational, non-command* speech from many speakers and
  recording conditions. A sample of it, labeled **UNKNOWN**, directly populates the
  empty 32nd class and gives the model real "not a command" examples to reject. It also
  provides the external OOD clips Part 8 needs to *measure* rejection (confidence
  threshold sweep).
- **Secondary:** its diverse accents and real recording conditions help D5 and D4.
- **Priority:** **P0** — the only dataset that fixes the empty UNKNOWN class.

### 4.2 Synthetic Speech Commands / SynTTS-Commands — **addresses D2** (and D3)
- **Deficiency fixed:** **D2 — limited numerical/time-expression diversity.**
- **How:** TTS can generate the existing slot values in many phrasings and (if the
  ontology is expanded) *arbitrary* numbers — "set a timer for 5 minutes", "set the
  temperature to 24 degrees", "set the brightness to 40 percent", "set an alarm for
  7:30". This trains the model to discriminate numbers generally, attacking the
  TEMPERATURE_22→18 / BRIGHTNESS_20→100 / TIMER_10→30 confusions.
- **Secondary:** also adds phrasing variety (D3).
- **Caveat:** synthetic speech is acoustically narrower than human speech; use it to
  *augment* real data, not replace it.
- **Priority:** **P1** — most flexible fix for the number-discrimination gap.

### 4.3 Timers and Such — **addresses D2** (timer-specific) and D3
- **Deficiency fixed:** **D2 — timer numerical diversity** (and D3 — timer phrasing).
- **How:** ~1,000 real human utterances of timer commands ("set a timer for 5 minutes",
  "set a timer for 10 seconds", …). Directly adds real, diverse timer phrasings and
  numbers to the TIMER family, which is currently 3 values × 3 templates and shows
  TIMER_10→30 confusion.
- **Priority:** **P1** — targeted, real-speech fix for the TIMER slot family.

### 4.4 SLURP — **addresses D3** (full-command phrasing) and D2 (diverse numbers)
- **Deficiency fixed:** **D3 — limited full-command phrasing** (and D2 — diverse slot numbers).
- **How:** ~10,000 real utterances across 100 commands in 10 domains (media, lighting,
  timer, alarm, phone, …) as full sentences ("turn on the lights", "set a timer for 5
  minutes", "what's the weather"). Adds many real phrasings for the full-command
  intents (TIMER, ALARM, LIGHT_*, PLAY_MUSIC, WEATHER, TIME) and diverse numbers for
  the slotted ones.
- **Caveat:** SLURP's 100 commands do not 1:1 match the 31 canonical classes; a mapping
  (EXACT/SEMANTIC/EXCLUDE) is required, as in Part 3.
- **Priority:** **P2** — broadest real-speech phrasing diversity for full commands.

### 4.5 Fluent Speech Commands (FSC) — **addresses D3** (single-word commands)
- **Deficiency fixed:** **D3 — limited pronunciation variety for the weak single-word classes.**
- **How:** FSC's command classes are single words — **play, stop, pause, up, down, on,
  off** — spoken naturally by ~105 speakers. These map directly onto the weakest
  OptionB classes: STOP, PAUSE, PLAY_MUSIC, VOLUME_UP, VOLUME_DOWN, LIGHT_ON,
  LIGHT_OFF. Adds real pronunciation variety exactly where Part 8 shows the most
  single-word confusion (STOP, LIGHT_ON, PAUSE↔STOP, LIGHT_ON↔LIGHT_OFF).
- **Caveat:** FSC words are single tokens; they augment the *pronunciation* of an
  existing intent, not a new class.
- **Priority:** **P2** — targeted fix for the weak single-word media/light/volume classes.

### 4.6 Google Speech Commands v2 (GSC v2) — **addresses D3** (single words) and D5 (accents)
- **Deficiency fixed:** **D3 — single-word pronunciation variety** (and D5 — accent/speaker diversity).
- **How:** ~100,000 utterances, 109 speakers, diverse accents, of single words
  (**up, down, on, off, stop, go**, zero–nine). Reinforces the same weak single-word
  classes as FSC and broadens the accent envelope (D5). The zero–nine digits also add
  number variety (D2).
- **Priority:** **P3** — overlaps FSC; use if FSC alone is insufficient or to widen accents.

---

## 5. Priority matrix

| Priority | Dataset | Fixes | Why now |
|---------|---------|-------|---------|
| **P0** | Common Voice | D1 (UNKNOWN/OOD) | Empty 32nd class; rejection currently unmeasurable |
| **P1** | SynTTS-Commands (synthetic) | D2 (numbers) | Worst class (TEMPERATURE_22) is a number-discrimination failure |
| **P1** | Timers and Such | D2 (timer numbers) + D3 | Real, targeted fix for the TIMER slot family |
| **P2** | SLURP | D3 (full-command phrasing) + D2 | Broadest real phrasing variety for full commands |
| **P2** | Fluent Speech Commands | D3 (single words) | Directly targets STOP / LIGHT_ON / PAUSE / VOLUME weakness |
| **P3** | Google Speech Commands v2 | D3 (single words) + D5 (accents) | Overlaps FSC; optional accent widening |

**Not recommended (no deficiency):** adding data to increase total size, or to top up
any of the 31 in-distribution classes (all already have 528–600 recordings).

---

## 6. What this step did NOT do

- Did **not** download any dataset.
- Did **not** modify `manifest.csv` or any OptionB metadata.
- Did **not** add external recordings to the canonical dataset.
- Did **not** expand the 32-class ontology (flagged as a Part 10 design decision).
- Did **not** train a model.

---

## 7. Suggested next step (Part 10)

1. **Decide the rejection strategy:** (a) add Common Voice OOD clips as UNKNOWN and
   retrain a 32-class model, or (b) keep 31 classes and tune a softmax-confidence
   threshold. Either way, a small external OOD set is required to *measure* rejection.
2. **Decide the slot-value scope:** keep the 3 fixed values per slotted class (and
   improve discrimination with diverse phrasings via SynTTS / Timers and Such / SLURP),
   or expand the ontology to more slot values (changes the class count).
3. On approval, download the P0/P1 datasets first (Common Voice + SynTTS + Timers and
   Such), map them with the same EXACT/SEMANTIC/EXCLUDE discipline as Part 3, and
   re-validate before any retraining.
