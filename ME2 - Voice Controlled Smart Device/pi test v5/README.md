# pi test v5 — ME2 voice command listener **with wake word + spoken responses**

The full interactive loop:

```
STANDBY: wait for "hey rhasspy"  →  play "yes?"  →  wait for a command  →
run the model  →  classify  →  play the matching TTS wav  →  back to STANDBY
        (repeat until Ctrl-C)
```

A **wake word** ("hey rhasspy", openWakeWord) gates the whole pipeline. The
grammar-constrained decoder no longer runs continuously — it only arms after
the wake word fires, which is what kills the *ghost commands* a noisy mic used
to produce (noise can no longer decode into a command on its own). The wake
word is **mandatory**: if the model can't load (missing file, unfetched
Git-LFS pointer, failed self-test), the program **refuses to start** with a
loud error instead of silently falling back to always-listen. Pass
`--no-wake` for always-listen, **debugging only**.

## The model

The **best model** from the vcm-v2 work — the **PocketSphinx ensemble**:

* **custom** 1.6 MB LDA AM (trained on the ME2 dataset) **and**
* **stock** 6.4 MB `en-us` AM

decode the *same* 103-phrase JSGF command grammar; the final command is chosen
by **agreement**, or by the **more confident stage-2 classifier** when they
disagree. **95.3% command / 97.1% intent** on the 171-clip held-out set
(`data/additional_test_data`, one new speaker) — see
`archived/vcm-v2/backbone/reports/pocketsphinx_ensemble_cmudict.json` (171
clips) and `test_v5_report.json` (176 clips incl. 5 REJECT, per-folder).

No Whisper / ONNX / torch — just `pocketsphinx` + `scikit-learn` + `numpy` +
`openwakeword`, so it runs comfortably on a Pi.

## The wake word

| | |
|---|---|
| Model | openWakeWord **"hey rhasspy"** (204 KB ONNX, bundled in `wakeword/`) |
| Threshold | 0.5 (phrase peaks ~0.8–0.9; real mic noise scores ~0.002) |
| Window | rolling 5 × 80 ms frames (0.4 s), peak taken over the window |
| After fire | plays `responses/00_yes.wav` ("yes?"); the mic is **muted while it plays**, so the wake word's own tail can't be decoded as a command |
| Command window | 1.0 s after the cue for speech to *start* (webrtcvad; 0.6 s of silence ends the utterance) |
| Cooldown | 0.5 s after the response, back to STANDBY |
| Self-test | `python vcm_pi_v5.py --wake-check` feeds the bundled fixture (`wakeword/selftest_hey_rhasspy.wav`) through the model and reports the exact failing step if it can't pass |
| Self-heal | if the on-disk model is an unfetched Git-LFS pointer (sub-KB), it is auto-downloaded from GitHub's raw endpoint on startup (needs internet once); `git lfs pull` works too |

**Dependency note:** `openwakeword` is pinned to **0.4.0** in
`requirements.txt`. Versions 0.5/0.6 require `tflite-runtime`, which has no
wheel for Python 3.13 / aarch64 (the Pi's environment) and will not install.

## Files

| Path | What it is | Size |
|---|---|---|
| `vcm_pi_v5.py` | the listener: wake word → "yes?" → VAD → ensemble → classify → **play wav** → loop | 16 KB |
| `wakeword/hey_rhasspy_v0.1.onnx` | openWakeWord "hey rhasspy" model (Git LFS; auto-heals if unfetched) | 204 KB |
| `wakeword/selftest_hey_rhasspy.wav` | fixture for `--wake-check` | — |
| `am/custom/` | custom 1.6 MB LDA acoustic model (+ its `vcm.lm.bin`) | ~1.7 MB |
| `am/stock/` | stock `en-us` acoustic model | ~6.4 MB |
| `am/stock_enus/` | stock `cmudict-en-us.dict` + `en-us.lm.bin` (from the pocketsphinx package) | ~29 MB |
| `dict3` | word→phone dictionary for the custom AM | 1.2 KB |
| `vcm_commands_enh3.jsgf` | the 103-phrase command grammar | 2.9 KB |
| `classifier.pkl` | stage-2 text classifier (31 commands + REJECT) | 5.2 MB |
| `responses/` | the 19 TTS response WAVs + `00_yes.wav` cue + generated `19_repeat.wav` | ~1.4 MB |
| `vcm/`, `vcm2/` | self-contained code (normalization, classifier, ground truth) | — |
| `test_wake_flow.py` | wake-word gate test (synthetic 48 kHz stream through the Pi path) | — |
| `test_v5_report.json` | 176-clip held-out eval (per-folder breakdown) | — |
| `requirements.txt` | deps (`openwakeword==0.4.0` pinned) | — |
| `training/` | AM training + eval scripts (`build_trained_am.py`, `eval_pocketsphinx*.py`) | — |

## Response mapping (31 commands → 19 WAVs)

The TTS repo has **19** response phrases; the 31 fine-grained commands map onto
them (several commands share a phrase). `REJECT` / unknown → **`19_repeat.wav`**
("can you repeat that?", generated with Piper `en_US-lessac-medium`).

| WAV | Says | Commands |
|---|---|---|
| `01_playing_music` | playing music | PLAY_MUSIC |
| `02_current_weather` | here's the current weather | WEATHER |
| `03_current_time` | here's the current time | TIME |
| `04_switching_lights` | switching the lights | LIGHT_ON, LIGHT_OFF |
| `05_pausing` | pausing | PAUSE |
| `06_stopping_playback` | stopping playback | STOP |
| `07_next_song` | playing the next song | NEXT |
| `08_volume_up` | increasing the volume | VOLUME_UP |
| `09_volume_down` | decreasing the volume | VOLUME_DOWN |
| `10_calling` | calling | CALL |
| `11_sending_message` | sending a message | MESSAGE |
| `12_reminders_list` | here are your reminders | LIST_REMINDERS |
| `13_setting_timer` | setting a timer | TIMER_10s/30s/1m |
| `14_setting_alarm` | setting alarm | ALARM_6_00AM/8_00AM/9_00PM |
| `15_changing_temperature` | changing temperature | TEMPERATURE_18/22/26 |
| `16_setting_brightness` | setting brightness | BRIGHTNESS_20/60/100 |
| `17_changing_color` | changing color | COLOR_RED/GREEN/BLUE |
| `18_creating_reminder` | creating reminder | CREATE_REMINDER_* |
| `19_repeat` | can you repeat that? | **REJECT** (unknown / no command) |

(`00_yes.wav` — "yes?" — is kept as a spare generic ack, not bound to a command.)

## Install (on the Pi)

```bash
pip install -r requirements.txt
# mic: usually works out of the box; for USB mics check `arecord -l`
```

## Run

```bash
python vcm_pi_v5.py --wake-check          # self-test the wake model (do this first)
python vcm_pi_v5.py                       # live mic: STANDBY → "hey rhasspy" → "yes?" → command → speak
python vcm_pi_v5.py --file clip.wav       # classify one file, print + play the response
python vcm_pi_v5.py --test                # run the held-out set (data/additional_test_data)
python vcm_pi_v5.py --no-play             # (mic) classify + print, skip playback
python vcm_pi_v5.py --wake-threshold 0.6  # stricter wake-word gate
python vcm_pi_v5.py --command-window 1.5  # wait 1.5 s for the command
python vcm_pi_v5.py --no-wake             # always-listen — DEBUGGING ONLY
```

Stop with **Ctrl-C** at any time.

### Per-utterance output (live mode)

```
[14:32:07] STANDBY — say "hey rhasspy" ...
[14:32:09] wake word (score 0.87) → playing 00_yes.wav
[14:32:11] ── utterance #3 (1.12 s) ─────────────────────
  transcript : 'turn on the lights'
  command    : LIGHT_ON  (prob 0.97)   intent: lights_switch
  E2E 92 ms
  >> playing 04_switching_lights.wav
[14:32:12] STANDBY — say "hey rhasspy" ...
```

## Latency (Pi 5, CPU)

The ensemble decodes each utterance with two small AMs — roughly
**~90–120 ms** end-to-end per command (the eval reports `asr_ms_p50 ≈ 87 ms`),
plus the response playback.

### Decoder warm-up (why the first command is re-decoded)

A freshly-created PocketSphinx decoder needs a few **real-speech** utterances to
lock in (its feature state adapts over the first decodes). A cold decoder can
mis-decode the *first* command. `vcm_pi_v5.py` handles this with no bundled
audio: it **re-decodes the first real command `WARMUP_REPS` (4) times and takes
the converged result**, which both fixes that command and warms the decoder for
all the ones that follow. (Warming with the Piper TTS voice was tried and
*rejected* — a different speaker biases the decoder.)

## Notes

* Audio is **not** committed (ME2 policy); `--test` reads
  `../data/additional_test_data` (the held-out set lives in the ME2
  `data/` folder).
* The response WAVs are 16 kHz mono 16-bit — the same format as the mic path,
  so no conversion is needed to play them.
* The wake model (`wakeword/hey_rhasspy_v0.1.onnx`) is tracked via **Git LFS**.
  If the file on the Pi is a sub-KB pointer, the program auto-downloads the
  real model from GitHub on startup (internet needed once); `git lfs pull`
  works too. Verify with `python vcm_pi_v5.py --wake-check`.
* Known accuracy gap: a few confusable pairs still slip through (e.g.
  **TIME → PLAY_MUSIC**). Candidate fixes: grammar/LM tuning or reweighting
  the stage-2 classifier on the confusion pairs.
