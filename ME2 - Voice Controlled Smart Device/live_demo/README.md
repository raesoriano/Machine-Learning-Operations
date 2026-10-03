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

v3 and v7 are the same script with different acoustic-model weights; the
adapter rolls their 31 fine classes up to the 19-command schema so the shared
loop + response map drive all models identically.

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

## Responses

`responses/` holds the 19 TTS WAVs (Piper `en_US-lessac-medium`, 16 kHz mono)
+ the "yes?" cue. The 19 coarse commands map onto these; REJECT/unknown plays
"can you repeat that?". Dynamic answers (time, weather) are synthesized with
`piper-tts` at runtime.

## Layout

```
live_demo/
├── vcm_live.py          # the entry point (wake word, VAD, music, TTS, mic loop)
├── shared.py            # shared infra + model factory + 19-intent response map
├── adapters/            # one adapter per model family (v3/v7, v8)
├── responses/           # 19 TTS WAVs + "yes?" cue
├── wakeword/            # "hey rhasspy" openWakeWord ONNX
├── vcm/ vcm2/           # PocketSphinx ensemble code (v3/v7 only)
└── requirements.txt
```
