# pi test v3 — ME2 voice command listener **with wake word + spoken responses**

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
clips) and `test_v3_report.json` (176 clips incl. 5 REJECT, per-folder).

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
| Self-test | `python vcm_pi_v3.py --wake-check` feeds the bundled fixture (`wakeword/selftest_hey_rhasspy.wav`) through the model and reports the exact failing step if it can't pass |
| Self-heal | if the on-disk model is an unfetched Git-LFS pointer (sub-KB), it is auto-downloaded from GitHub's raw endpoint on startup (needs internet once); `git lfs pull` works too |

**Dependency note:** `openwakeword` is pinned to **0.4.0** in
`requirements.txt`. Versions 0.5/0.6 require `tflite-runtime`, which has no
wheel for Python 3.13 / aarch64 (the Pi's environment) and will not install.

## Files

| Path | What it is | Size |
|---|---|---|
| `vcm_pi_v3.py` | the listener: wake word → "yes?" → VAD → ensemble → classify → **play wav / drive the music player / speak a live answer** → loop | 24 KB |
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
| `mpv.log` | mpv's output (created at runtime when music plays; track-load errors land here) | — |
| `test_v3_report.json` | 176-clip held-out eval (per-folder breakdown) | — |
| `requirements.txt` | deps (`openwakeword==0.4.0` pinned) | — |
| `training/` | AM training + eval scripts (`build_trained_am.py`, `eval_pocketsphinx*.py`) | — |

## Response mapping (31 commands → 19 WAVs)

The TTS repo has **19** response phrases; the 31 fine-grained commands map onto
them (several commands share a phrase). `REJECT` / unknown → **`19_repeat.wav`**
("can you repeat that?", generated with Piper `en_US-lessac-medium`).

| WAV | Says | Commands |
|---|---|---|
| `01_playing_music` | playing music | PLAY_MUSIC ² |
| `02_current_weather` | here's the current weather | WEATHER ¹ |
| `03_current_time` | here's the current time | TIME ¹ |
| `04_switching_lights` | switching the lights | LIGHT_ON, LIGHT_OFF |
| `05_pausing` | pausing | PAUSE ² |
| `06_stopping_playback` | stopping playback | STOP ² |
| `07_next_song` | playing the next song | NEXT ² |
| `08_volume_up` | increasing the volume | VOLUME_UP ² |
| `09_volume_down` | decreasing the volume | VOLUME_DOWN ² |
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

¹ **Dynamic:** WEATHER and TIME ignore the canned WAV and speak a live answer
via Piper TTS — see [Dynamic responses](#dynamic-responses-time-weather).

² **Real playback:** the six media commands (PLAY_MUSIC, PAUSE, STOP, NEXT,
VOLUME_UP, VOLUME_DOWN) ignore the canned WAV and actually control a YouTube
playlist via **mpv + yt-dlp**, then speak a live confirmation — see
[Music (YouTube playlist)](#music-youtube-playlist).

## Getting just this folder (sparse checkout)

The full repo is large (data + archived models). On the Pi, clone only this
folder:

```bash
git clone --filter=blob:none --sparse https://github.com/raesoriano/Machine-Learning-Operations.git
cd Machine-Learning-Operations
git sparse-checkout set "ME2 - Voice Controlled Smart Device/pi test v3"
```

(Already have a full clone? Just `cd "ME2 - Voice Controlled Smart Device/pi test v3"`
— no need to re-clone.)

## Install (on the Pi)

```bash
cd "ME2 - Voice Controlled Smart Device/pi test v3"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # THIS folder's requirements — torch-free
# music (the six media commands) needs mpv — a SYSTEM package, not pip:
sudo apt install mpv
# mic: usually works out of the box; for USB mics check `arecord -l`
```

> **Do NOT install PyTorch on the Pi.** The repo-root `requirements.txt` (and
> `setup_env.sh`) is the GPU *training* stack for ME1 / the old CTC model — it
> pulls in `torch`, `torchvision`, `torchaudio`, `transformers`, `lightning`,
> `gradio`, etc. (a ~2 GB footprint). The current model (`pi test v3`) needs
> none of that: it runs on `pocketsphinx` + `scikit-learn` + `openwakeword`
> only. Always install from **this folder's** `requirements.txt`, never the
> repo-root one, and never run `bash setup_env.sh` on the Pi.

## Run

```bash
python vcm_pi_v3.py --wake-check          # self-test the wake model (do this first)
python vcm_pi_v3.py                       # live mic: STANDBY → "hey rhasspy" → "yes?" → command → speak
python vcm_pi_v3.py --file clip.wav       # classify one file, print + play the response
python vcm_pi_v3.py --test                # run the held-out set (data/additional_test_data)
python vcm_pi_v3.py --no-play             # (mic) classify + print, skip playback
python vcm_pi_v3.py --wake-threshold 0.6  # stricter wake-word gate
python vcm_pi_v3.py --command-window 1.5  # wait 1.5 s for the command
python vcm_pi_v3.py --no-wake             # always-listen — DEBUGGING ONLY
python vcm_pi_v3.py --weather-loc "UP Diliman, Quezon City"  # pin the WEATHER location
python vcm_pi_v3.py --playlist "https://www.youtube.com/playlist?list=PL..."  # change the playlist
python vcm_pi_v3.py --volume 75            # start the music at 75 %
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

## Dynamic responses (TIME, WEATHER)

Two commands do **not** play a canned WAV — they synthesize a live answer with
Piper TTS (the same `en_US-lessac-low` voice used for `19_repeat.wav`):

| Command | What it does |
|---|---|
| `TIME` | Says the actual current time (UTC+8, Asia/Manila): *"It is 5:42 PM."* |
| `WEATHER` | Looks up the device's **current location** from its public IP (`ip-api.com`, with `ipinfo.io` as a fallback; both free, no key), fetches the current conditions from **Open-Meteo** (free, no key), and speaks them: *"Currently in Quezon City: partly cloudy, 29 degrees Celsius, feels like 33."* The spoken name is **city-level** (e.g. "Quezon City", "Pasig") — never a barangay, because free IP-geolocation databases are unreliable below city level. |

**Location fallback:** if the IP lookup fails (offline, blocked, private IP),
the weather is fetched for **UP Diliman, Quezon City** instead and the device
still answers. If the weather service itself is unreachable, it says
*"I could not reach the weather service right now."* — the lookup is bounded
(10 s per request) so a dead network never hangs the device.

To pin a location name (e.g. while testing away from home):

```bash
python vcm_pi_v3.py --weather-loc "UP Diliman, Quezon City"
```

## Music (YouTube playlist)

The six media commands — `PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT`, `VOLUME_UP`,
`VOLUME_DOWN` — do **not** just play an acknowledgement WAV. They actually
control a **YouTube playlist**, and then speak a live confirmation with Piper
TTS.

| Command | What it does | Spoken |
|---|---|---|
| `PLAY_MUSIC` | Starts the playlist (resumes if paused) | "playing the playlist" / "resuming the playlist" |
| `PAUSE` | Pauses playback | "paused" |
| `STOP` | Pauses and rewinds to the start of the current track | "stopped" |
| `NEXT` | Skips to the next track in the playlist | "next song" |
| `VOLUME_UP` | Steps volume **up** one level | "volume 75 percent" |
| `VOLUME_DOWN` | Steps volume **down** one level | "volume 50 percent" |

**Volume is discrete:** it moves in fixed steps through
**0 → 25 → 50 → 75 → 100 %** (starting at 50 % by default, or whatever
`--volume` sets). It never goes above 100 % or below 0 %.

**How it works:** **yt-dlp** first resolves the YouTube playlist into a
plain `playlist.m3u` next to the script (fast, flat — no per-track
metadata). **mpv** (a headless command-line player) then plays that file,
with each YouTube track loaded through its bundled ytdl hook (which also
uses yt-dlp). mpv runs as a separate process with a JSON IPC socket, so
each voice command is a one-line message over that socket and playback
never blocks the mic loop. Resolving the playlist up front is deliberate:
on a Pi, mpv handed a raw YouTube *playlist* URL often loads nothing,
which is why the old version could say "playing the playlist" while the
speaker stayed silent. The default playlist is the one built into the
code; change it with `--playlist`.

**Verify it before going live (no mic needed):**

```bash
python vcm_pi_v3.py --music-test
```

This resolves the playlist, starts mpv, plays ~12 s, skips a track,
raises the volume, and stops. If you hear music, the six voice commands
will work.

**Setup (once, on the Pi):**

```bash
sudo apt install mpv        # the player (system package, NOT a pip package)
pip install yt-dlp          # already in this folder's requirements.txt
```

**Graceful degradation:** if `mpv` or `yt-dlp` is missing, or the playlist
cannot be resolved (offline / playlist gone), the music commands say
*"music is not available right now"* (or *"nothing is playing"*) instead
of crashing — the rest of the device (lights, weather, time, …) keeps
working. mpv's own output is logged to `mpv.log` next to the script if a
track fails to load.

## Latency (Pi 5, CPU)

The ensemble decodes each utterance with two small AMs — roughly
**~90–120 ms** end-to-end per command (the eval reports `asr_ms_p50 ≈ 87 ms`),
plus the response playback.

### Decoder warm-up (why the first command is re-decoded)

A freshly-created PocketSphinx decoder needs a few **real-speech** utterances to
lock in (its feature state adapts over the first decodes). A cold decoder can
mis-decode the *first* command. `vcm_pi_v3.py` handles this with no bundled
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
  works too. Verify with `python vcm_pi_v3.py --wake-check`.
* Known accuracy gap: a few confusable pairs still slip through (e.g.
  **TIME → PLAY_MUSIC**). Candidate fixes: grammar/LM tuning or reweighting
  the stage-2 classifier on the confusion pairs.
