# ME2 dataset

The current ME2 dataset is **not stored in this repo** (it is ~7 GB of 16 kHz
WAVs). It lives on Hugging Face:

> **[`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)**

This folder holds the **description** of the data and a **one-command download
script** that builds the layout the v8 training/eval code expects.

## What the data is

A voice-command dataset for a smart device: **19 command intents** (play music,
weather, time, lights on/off, brightness, color, timer, alarm, temperature,
pause/stop/next, volume up/down, call, message, list reminders, create
reminder) with **93 phrase variations** and **18 slot values** (times, percents,
temperatures, reminder texts), plus an **out-of-scope** class for rejection.

### Splits

| split | clips | role |
|---|---:|---|
| `train` | 10,733 | in-scope command clips (real + synthetic voices) |
| `test` | 4,443 | held-out, **121 speakers, zero overlap with train** (4,367 in-scope + 76 OOS) |
| `holdout` | 202 | class holdout used by the benchmark (93 variations × 2 + 16 OOS) |
| `numerals` | 66,390 | number-only clips (MLEnd + Speech Commands digits) — separate from the splits, used to teach the slot values |
| `synthetic_negatives` | 1,250 | generated out-of-scope clips (train 1,000 / test 250) — rejection training/eval |

**81,818 clips / ~30.9 h** in the default config. 315 train speakers, 121 test
speakers (disjoint). The test split is ~81 % synthetic — the dataset audit
therefore reports **real and synthetic scores separately** so synthetic voices
cannot mask real-speech performance.

### Sources (each keeps its own license)

Speech Commands v2, MLEnd numerals, group synthetic TTS, SLURP, group real
voices, SNIPS SLU, Fluent Speech Commands, Common Voice en, Xela S1–S5,
Timers and Such, Xela Multi-Sensor. See the HF card for the full per-source
license table (SLURP CC-BY-4.0, Speech Commands CC-BY-4.0, etc.).

## Columns

`audio` (16 kHz waveform) · `file` · `transcript` · `command` (coarse 19
schema or `OUT_OF_SCOPE`) · `variation` · `slot_value` · `out_of_scope` ·
`bucket` · `speaker_id` · `source` · `is_synthetic` · `accent_group` ·
`numerals` · `duration_s` · `transcript_source` · `variation_match` ·
`whisper_check` · `whisper_transcript` · `note`.

## Download (one command)

```bash
cd "ME2 - Voice Controlled Smart Device/data"
python download_dataset.py            # -> ./dataset/{train,test,holdout,numerals}/{audio,manifest.csv}
```

This writes the layout `train_v8.py` / `eval_v8.py` read by default
(`--data ./dataset`). It needs `datasets` + `pandas` + `soundfile`
(`pip install datasets pandas soundfile`). ~7 GB, a few minutes.

The v8 code's default `--data` path points at this folder's `dataset/`
subdirectory; pass `--data <path>` to override.

## .gitignore

Everything under `data/` is ignored except this README and the download script
— the audio never enters git.
