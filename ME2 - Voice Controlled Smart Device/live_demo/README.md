# ME2 live demo — reusable live-mic voice command listener

One script, `vcm_live.py`, runs the full live loop with a **swappable model**:

```
STANDBY  "hey rhasspy" (openWakeWord, always on)
   │  score ≥ 0.5 over a 0.4 s window
   ▼
CUE     play "yes?" (mic muted while it plays, so the wake word's tail
        can never be decoded as a command)
   ▼
COMMAND wait up to 1.0 s for speech to start (webrtcvad; 0.6 s of silence
        ends the utterance) → decode → classify → play the matching TTS response
   ▼
STANDBY 0.5 s cooldown, then wait for the next wake word
```

The wake word is **mandatory**: if the model can't load (missing file,
unfetched LFS pointer, failed self-test) the script **refuses to start**
instead of silently falling back to always-listen. `--no-wake` opts into
always-listen for debugging only.

## Models

| `--model` | recognizer | needs |
|---|---|---|
| `v8` (default) | Conformer+CTC **ONNX** — `pi test v8-conformer-ctc/models/best.onnx` (736-vocab, content reject) | `onnxruntime` + `numpy` only |
| `v8` + `--v8-model ../pi\ test\ v8-conformer-ctc/models_neg/best.onnx --no-reject-content --no-reject-empty` | old 109-vocab + `--reject-empty` | same |
| `v3` | PocketSphinx ensemble (original AM) — `archived/pi test v3` | + `pocketsphinx`, `scikit-learn` |
| `v7` | PocketSphinx ensemble (AM retrained on v6 data) — `archived/pi test v7` | + `pocketsphinx`, `scikit-learn` |

v3 and v7 are the same script with different acoustic-model weights.

## The 31 commands

The command space is **31 commands = 13 fixed + 6 slotted × 3 values (18)**:

| group | commands |
|---|---|
| **13 fixed** (same action regardless) | `play_music`, `weather`, `time`, `light_on`, `light_off`, `pause`, `stop`, `next`, `volume_up`, `volume_down`, `call`, `message`, `list_reminders` |
| **6 slotted** (each has 3 values) | `timer` (10 s / 30 s / 1 min), `alarm` (6 AM / 8 AM / 9 PM), `temperature` (18 / 22 / 26 °), `brightness` (20 / 60 / 100 %), `color` (red / green / blue), `create_reminder` (drink water / study / exercise) |

13 + 18 = **31**. Each command has **3 phrase variations** → **93 phrases**
(see `pi test v8-conformer-ctc/variations.csv`). Anything that is not one of
the 31 is **REJECTED** ("can you repeat that?").

v8 emits the **fine** class (e.g. `ALARM_6_00AM`), which carries the slot; v3/v7
emit the coarse label and the slot is recovered from the decoded transcript.

## Setup (Raspberry Pi 5)

```bash
cd "ME2 - Voice Controlled Smart Device/live_demo"
python3 -m venv .venv && source .venv/bin/activate

# v8 only (lightest — no torch):
pip install -r requirements.txt        # onnxruntime + audio + wake word + TTS
# for v3 / v7 too: uncomment pocketsphinx + scikit-learn in requirements.txt

sudo apt install -y mpv                # music commands (playlist playback)
```

The v8 ONNX + wake-word ONNX are already in the sparse checkout (LFS). If the
v8 weights aren't present, also sparse-checkout
`ME2 - Voice Controlled Smart Device/pi test v8-conformer-ctc` and run
`git lfs pull`.

## Run

```bash
python vcm_live.py --model v8                 # LIVE MIC (v8, current model)
python vcm_live.py --model v8 --file clip.wav # classify one file, print + speak
python vcm_live.py --model v3                 # PocketSphinx ensemble (v3 AM)
python vcm_live.py --model v7                 # PocketSphinx ensemble (v7 AM)
```

Useful flags:

| flag | effect |
|---|---|
| `--no-play` | classify + print, skip TTS playback |
| `--no-wake` | always-listen (debugging only) |
| `--wake-threshold 0.6` | stricter wake-word gate |
| `--command-window 1.5` | wait 1.5 s for the command |
| `--gate` | extra post-decode confidence gate |
| `--playlist <url>` | YouTube playlist for the music commands |
| `--volume 50` | start volume (0–100) |
| `--weather-loc "Manila"` | city for the weather command |
| `--music-test` | stub-mpv self-test of the media commands |
| `--wake-check` | verify the wake word loads, then exit |

## Responses (Piper TTS)

There are **no pre-recorded command-response WAVs** anymore. Every command
response is synthesized live with **Piper TTS** (`piper-tts`, the
`en_US-lessac-low` voice — the fastest/lightest tier, ~63 MB, auto-downloaded
into `tts/` on first run). The spoken text carries the **slot** for slotted
intents, e.g. *"Setting an alarm for 6:00 AM."* (not just "setting alarm").
REJECT → *"Can you repeat that?"*. `time` and `weather` are answered
dynamically (live clock / live weather).

The only pre-recorded audio is `responses/00_yes.wav` — the "yes?" cue played
after the wake word (a prompt, not a command response).

## Layout

```
live_demo/
├── vcm_live.py          # entry point (wake word, VAD, music, Piper TTS, mic loop)
│                        #   + response_text(): 31 commands + REJECT -> spoken text
├── shared.py            # shared infra + model factory
├── adapters/            # one adapter per model family (v3/v7, v8)
├── responses/           # 00_yes.wav ("yes?" cue) only
├── tts/                 # Piper voice (auto-downloaded, git-ignored)
├── wakeword/            # "hey rhasspy" openWakeWord ONNX + self-test wav
├── vcm/ vcm2/           # PocketSphinx ensemble code (v3/v7 only)
└── requirements.txt
```
