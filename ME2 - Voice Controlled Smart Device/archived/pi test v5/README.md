# pi test v5 — lightweight ONNX voice-command listener

The same device loop as v3/v4, but with a **tiny** model that actually reports
what it heard and can genuinely reject.

```
"hey rhasspy"  ->  VAD capture  ->  limited-vocab ASR (what was HEARD)
                ->  print + speak the heard words
                ->  phrase classifier: 31 commands  OR  REJECT
                ->  execute (music / weather / time / lights)  or
                    "I heard <words>. Can you repeat that?"
```

## Why v5 (vs v3 and v4)

| | v3 | v4 | **v5** |
|---|---|---|---|
| transcript | grammar phrase (not what you said) | whisper free text | **limited-vocab words (what was heard)** |
| rejection | none (forced to a command) | TF-IDF gate | **learned REJECT class** |
| ASR model | PocketSphinx (8 MB) | whisper base.en (73 MB) | **1-D CNN (≈0.5 MB ONNX)** |
| classifier | TF-IDF + LR (5 MB pkl) | TF-IDF + LR | **1-D CNN (≈0.3 MB ONNX)** |
| new pip deps | pocketsphinx | faster-whisper (+ctranslate2, ~300 MB) | **onnxruntime only** |
| model footprint | ~8 MB | ~80 MB + 300 MB deps | **~1 MB total** |
| latency (Pi 4) | ~140 ms | ~1–3 s | **~100–300 ms** |

The two problems v4's whisper was solving — *"the transcript is a grammar
phrase, not what I said"* and *"it never rejects, it force-fits"* — are solved
here with a small deep model instead of a 73 MB transformer:

1. **Limited-vocabulary ASR** (CTC, 1-D CNN): decodes the clip into words from
   the ~82-word command vocabulary. A word it can't match is emitted as
   `<unk>` — the model says what it actually heard, and admits when it doesn't
   know a word.
2. **Phrase classifier** (1-D CNN over the decoded words): maps the word
   sequence to one of the 31 commands **or REJECT**. Trained on the ASR's own
   decodes (so it sees the same errors it will get live) plus 2,400
   out-of-domain clips as the REJECT class.

## The models

| file | what | size |
|---|---|---|
| `models/asr.onnx` | limited-vocab CTC ASR (1-D CNN) | ~0.5 MB |
| `models/classify.onnx` | phrase classifier, 31 + REJECT (1-D CNN) | ~0.3 MB |
| `data/vocab.json` | 82-word vocabulary + class map | ~6 KB |

Feature pipeline (identical train / test / Pi): 16 kHz mono → 3.0 s window →
log-Mel (n_fft=400, hop=160, n_mels=40) → (40, 301). Implemented in pure
numpy/scipy (`feats.py`) so the Pi needs **no torch**.

## Run it (Pi)

```bash
cd ~/Machine-Learning-Operations
git pull
cd "ME2 - Voice Controlled Smart Device/pi test v5"
pip install -r requirements.txt        # onnxruntime + openwakeword + piper + yt-dlp
sudo apt install mpv                   # one-time, for the music commands

python vcm_pi_v5.py --wake-check       # expect PASS ~0.64
python vcm_pi_v5.py                    # live: say "hey rhasspy", then a command
```

Useful flags:

```bash
python vcm_pi_v5.py --file some.wav            # classify one clip, no mic
python vcm_pi_v5.py --test                     # run the 171-clip held-out set
python vcm_pi_v5.py --speak-heard              # also speak "I heard: ..."
python vcm_pi_v5.py --music-test               # verify the music pipeline
python vcm_pi_v5.py --no-play                  # print responses, don't play
```

## What you'll see

```
[22:08:40] ── utterance #7 (1.86 s) ──────────────────────────────
  heard      : 'play music'
  command    : PLAY_MUSIC   intent: play_music
  E2E 180 ms
  >> saying (Piper TTS): 'playing the playlist'
```

For something that isn't a command:

```
  heard      : 'turn the blender on'
  command    : REJECT   intent: reject
  >> saying (Piper TTS): "I heard turn the blender on. Can you repeat that?"
```

## Files

- `vcm_pi_v5.py` — the listener (wake word → VAD → ASR → classify → act).
- `feats.py` — numpy/scipy log-Mel (shared by training + Pi).
- `infer.py` — ONNX inference (the cascade; no torch).
- `models.py` — the two torch models (used only to train/export).
- `train_asr.py`, `train_classify.py` — training (HPC; not needed on the Pi).
- `data/prep_data.py` — builds the feature bundle from the OptionB manifest.
- `models/`, `data/` — the committed ONNX models + vocabulary.
- `wakeword/`, `responses/`, `vcm2/` — carried over from v3.
