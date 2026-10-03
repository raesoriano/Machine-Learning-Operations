# pi test v4 — ME2 voice command listener that **reports what was actually heard** and **rejects non-commands**

The full interactive loop:

```
STANDBY: wait for "hey rhasspy"  →  play "yes?"  →  wait for a command  →
whisper transcribes what was heard (printed)  →  keyword gate + classifier
→ a command or REJECT  →  speak the response (or "I heard: … can you repeat?")
        →  back to STANDBY  (repeat until Ctrl-C)
```

A **wake word** ("hey rhasspy", openWakeWord) gates the whole pipeline, exactly
as in v3 — noise can no longer produce a command on its own. The wake word is
**mandatory**: if the model can't load (missing file, unfetched Git-LFS
pointer, failed self-test), the program **refuses to start** with a loud error
instead of silently falling back to always-listen. Pass `--no-wake` for
always-listen, **debugging only**.

## What changed from v3 (and why)

v3's transcript came from a **grammar-constrained** PocketSphinx decoder
(JSGF). A constrained decoder is *forced* to output one of the 103 grammar
phrases, so two things went wrong:

1. **The transcript was not what you said** — it was the nearest grammar
   phrase. Say something that is not a command and it still printed
   "play music".
2. **The classifier's REJECT class was dead** — it was trained on 32,992
   out-of-domain examples to reject, but it only ever saw valid grammar
   phrases, so it force-classified everything into the nearest command.

v4 changes the front end to a real free-text ASR:

```
faster-whisper (base.en)  →  keyword gate  →  31-class + REJECT classifier
```

* **whisper** transcribes WHAT WAS ACTUALLY HEARD (free text, no grammar).
  The device prints it (`heard: 'he will do it my man'`) and, on a reject,
  SAYS it back: *"I heard: he will do it my man. Can you repeat that?"*
* **keyword gate** — a real command always contains a command word
  (play / lights / volume / weather / …). Speech with no command word is
  rejected immediately — the OOD rejection v3 could never produce.
* **classifier** — the same trained 31-class + REJECT model from v3, now fed
  real free text, so it can (and does) emit REJECT for the subtle non-commands
  the gate lets through.

### Held-out results (171-clip set, `data/additional_test_data`, one new speaker)

| | command | intent | OOD rejected | false-reject |
|---|---|---|---|---|
| v3 (constrained PocketSphinx) | 83.5% | 86.4% | **0/5** | 10 |
| **v4 (whisper + gate + classifier)** | **84.7%** | 84.7% | **4/5** | 21 |

The remaining v4 errors are almost all **whisper ASR mistakes on very short
words** ("pause" → "boss", "time" → "bye", "stop" → "and labor"). For those,
*"can you repeat that?"* is the honest answer — and exactly what this version
was asked to do (reject nonsense rather than misclassify). The one OOD leak
("mali mali naman sinasabi" → whisper heard "I will see you next time" → TIME)
is the same class of ASR error.

**Cost:** whisper `base.en` (int8) takes ~1–3 s per command on a Pi 4 (v3's
PocketSphinx was ~150 ms). The wake word + VAD are unchanged.

## The wake word

| | |
|---|---|
| Model | openWakeWord **"hey rhasspy"** (204 KB ONNX, bundled in `wakeword/`) |
| Threshold | 0.5 (phrase peaks ~0.8–0.9; real mic noise scores ~0.002) |
| Window | rolling 5 × 80 ms frames (0.4 s), peak taken over the window |
| After fire | plays `responses/00_yes.wav` ("yes?"); the mic is **muted while it plays**, so the wake word's own tail can't be decoded as a command |
| Command window | 1.0 s after the cue for speech to *start* (webrtcvad; 0.6 s of silence ends the utterance) |
| Cooldown | 0.5 s after the response, back to STANDBY |
| Self-test | `python vcm_pi_v4.py --wake-check` feeds the bundled fixture (`wakeword/selftest_hey_rhasspy.wav`) through the model and reports the exact failing step if it can't pass |
| Self-heal | if the on-disk model is an unfetched Git-LFS pointer (sub-KB), it is auto-downloaded from GitHub's raw endpoint on startup (needs internet once); `git lfs pull` works too |

**Dependency note:** `openwakeword` is pinned to **0.4.0** in
`requirements.txt`. Versions 0.5/0.6 require `tflite-runtime`, which has no
wheel for Python 3.13 / aarch64 (the Pi's environment) and will not install.

## Files

| Path | What it is | Size |
|---|---|---|
| `vcm_pi_v4.py` | the listener: wake word → "yes?" → VAD → **whisper** → **keyword gate** → **classifier** → **play wav / drive the music player / speak a live answer (incl. "I heard: …")** → loop | ~25 KB |
| `wakeword/hey_rhasspy_v0.1.onnx` | openWakeWord "hey rhasspy" model (Git LFS; auto-heals if unfetched) | 204 KB |
| `wakeword/selftest_hey_rhasspy.wav` | fixture for `--wake-check` | — |
| `classifier.pkl` | stage-2 text classifier (31 commands + REJECT) — same model as v3 | 5.2 MB |
| `responses/` | the 19 TTS response WAVs + `00_yes.wav` cue + generated `19_repeat.wav` | ~1.4 MB |
| `vcm/`, `vcm2/` | self-contained code (normalization, classifier, ground truth) | — |
| `test_wake_flow.py` | wake-word gate test (synthetic 48 kHz stream through the Pi path) | — |
| `mpv.log` | mpv's output (created at runtime when music plays; track-load errors land here) | — |
| `test_v4_report.json` | 176-clip held-out eval (per-folder breakdown) — written by `--test` | — |
| `requirements.txt` | deps (`openwakeword==0.4.0` pinned, `faster-whisper>=1.0.0`) | — |

The whisper model (`base.en`, ~145 MB) is **not** in this folder — it is
downloaded from Hugging Face on first run and cached in
`~/.cache/huggingface` (needs internet once).

## Response mapping (31 commands → 19 WAVs)

The TTS repo has **19** response phrases; the 31 fine-grained commands map onto
them (several commands share a phrase). `REJECT` → the device **says what it
heard** + "can you repeat that?" (Piper TTS, synthesized live) — it no longer
plays a canned "repeat" WAV, because the whole point of v4 is to report the
actual transcript.

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
| `19_repeat` | can you repeat that? | **REJECT** (now spoken live as "I heard: … can you repeat that?") |

(`00_yes.wav` — "yes?" — is kept as a spare generic ack, not bound to a command.)

¹ **Dynamic:** WEATHER and TIME ignore the canned WAV and speak a live answer
via Piper TTS — see [Dynamic responses](#dynamic-responses-time-weather).

² **Real playback:** the seven media commands (PLAY_MUSIC, PAUSE, STOP, NEXT,
PREVIOUS, VOLUME_UP, VOLUME_DOWN) ignore the canned WAV and actually control a
YouTube playlist via **mpv + yt-dlp**, then speak a live confirmation — see
[Music (YouTube playlist)](#music-youtube-playlist).

## Getting just this folder (sparse checkout)

The full repo is large (data + archived models). On the Pi, clone only this
folder:

```bash
git clone --filter=blob:none --sparse git@github.com:raesoriano/Machine-Learning-Operations.git
cd Machine-Learning-Operations
git sparse-checkout set "ME2 - Voice Controlled Smart Device/pi test v4"
```

(Already have a full clone? Just `cd "ME2 - Voice Controlled Smart Device/pi test v4"`
— no need to re-clone.)

## Install (on the Pi)

```bash
cd "ME2 - Voice Controlled Smart Device/pi test v4"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # THIS folder's requirements — torch-free
# music (the seven media commands) needs mpv — a SYSTEM package, not pip:
sudo apt install mpv
# mic: usually works out of the box; for USB mics check `arecord -l`
```

> **Do NOT install PyTorch on the Pi.** The repo-root `requirements.txt` (and
> `setup_env.sh`) is the GPU *training* stack for ME1 / the old CTC model — it
> pulls in `torch`, `torchvision`, `torchaudio`, `transformers`, `lightning`,
> `gradio`, etc. (a ~2 GB footprint). The current model (`pi test v4`) needs
> none of that: it runs on `faster-whisper` + `scikit-learn` + `openwakeword`
> only. Always install from **this folder's** `requirements.txt`, never the
> repo-root one, and never run `bash setup_env.sh` on the Pi.
>
> **First run downloads the whisper model** (`base.en`, ~145 MB) from Hugging
> Face into `~/.cache/huggingface` — the Pi needs internet once.

## Run

```bash
python vcm_pi_v4.py --wake-check          # self-test the wake model (do this first)
python vcm_pi_v4.py                       # live mic: STANDBY → "hey rhasspy" → "yes?" → command → speak
python vcm_pi_v4.py --file clip.wav       # transcribe + classify one file, print + play the response
python vcm_pi_v4.py --test                # run the held-out set (data/additional_test_data)
python vcm_pi_v4.py --no-play             # (mic) classify + print, skip playback
python vcm_pi_v4.py --wake-threshold 0.6  # stricter wake-word gate
python vcm_pi_v4.py --command-window 1.5  # wait 1.5 s for the command
python vcm_pi_v4.py --no-wake             # always-listen — DEBUGGING ONLY
python vcm_pi_v4.py --weather-loc "UP Diliman, Quezon City"  # pin the WEATHER location
python vcm_pi_v4.py --playlist "https://www.youtube.com/playlist?list=PL..."  # change the playlist
python vcm_pi_v4.py --volume 75            # start the music at 75 %
python vcm_pi_v4.py --whisper small.en     # bigger (slower) whisper model
```

Stop with **Ctrl-C** at any time.

### Per-utterance output (live mode)

```
[14:32:07] STANDBY — say "hey rhasspy" ...
[14:32:09] wake word detected → playing 00_yes.wav
[14:32:11] ── utterance #3 (1.12 s) ──────────────────────────
  heard      : 'turn on the lights'   (what was actually spoken)
  command    : LIGHT_ON   intent: lights_switch   (clf 0.97)
  E2E 1840 ms
  >> playing 04_switching_lights.wav
[14:32:13] STANDBY — say "hey rhasspy" ...
```

A non-command is reported and rejected:

```
  heard      : 'he will do it my man'   (what was actually spoken)
  command    : REJECT   intent: reject   (clf 0.00)
  >> saying (Piper TTS): "I heard: he will do it my man. Can you repeat that?"
```

## Dynamic responses (TIME, WEATHER)

Two commands do **not** play a canned WAV — they synthesize a live answer with
Piper TTS (the same `en_US-lessac-low` voice used for the reject line):

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
python vcm_pi_v4.py --weather-loc "UP Diliman, Quezon City"
```

## Music (YouTube playlist)

The seven media commands — `PLAY_MUSIC`, `PAUSE`, `STOP`, `NEXT`, `PREVIOUS`,
`VOLUME_UP`, `VOLUME_DOWN` — do **not** just play an acknowledgement WAV. They
actually control a **YouTube playlist**, and then speak a live confirmation
with Piper TTS.

| Command | What it does | Spoken |
|---|---|---|
| `PLAY_MUSIC` | Starts the playlist (resumes if paused) | "playing the playlist" / "resuming the playlist" |
| `PAUSE` | Pauses playback | "paused" |
| `STOP` | Pauses and rewinds to the start of the current track | "stopped" |
| `NEXT` | Skips to the next track in the playlist | "next song" |
| `PREVIOUS` | Skips back to the previous track | "previous song" |
| `VOLUME_UP` | Steps volume **up** one level | "volume 75 percent" |
| `VOLUME_DOWN` | Steps volume **down** one level | "volume 50 percent" |

**Volume is discrete:** it moves in fixed steps through
**0 → 25 → 50 → 75 → 100 %** (starting at 50 % by default, or whatever
`--volume` sets). It never goes above 100 % or below 0 %.

**"Previous" is routed from the heard text, not the classifier.** "previous
song" is not a trained classifier class (the 31-class model has no PREVIOUS),
so the live loop checks the heard text for "previous" and routes it to
PREVIOUS itself (it prints `(routed PREVIOUS from heard text)` when it does).
At a playlist boundary the device says "that's the first/last song in the
playlist" instead of skipping past the end.

**How it works:** **yt-dlp** first resolves the YouTube playlist into a
plain `playlist.m3u` next to the script (fast, flat — no per-track
metadata). **mpv** (a headless command-line player) then plays that file,
with each YouTube track loaded through its bundled ytdl hook (which also
uses yt-dlp). mpv runs as a separate process with a JSON IPC socket, so
each voice command is a one-line message over that socket and playback
never blocks the mic loop. The default playlist is the one built into the
code; change it with `--playlist`.

**Verify it before going live (no mic needed):**

```bash
python vcm_pi_v4.py --music-test
```

This resolves the playlist, starts mpv, plays ~12 s, skips a track,
raises the volume, and stops. If you hear music, the seven voice commands
work.

**Spoken responses (canned WAVs + Piper TTS):** responses are played by a
**separate-process player** (mpv, then paplay/aplay/ffplay as fallbacks),
not in-process `sounddevice`. On the Pi, opening an output stream with
`sd.play` while the mic input stream is open fails with `PortAudioError`
(the mic keeps working, but every response goes silent). A subprocess opens
its own audio stream — the same reason the mpv music playback works — so
responses and music now both play reliably. If no external player is found,
the code falls back to in-process `sd.play` (fine on headless dev machines
or when the mic is not open).
