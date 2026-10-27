# ME3-FullDuplex

A full-duplex version of ME 3: the agent **listens while it speaks** and
supports **barge-in** (interrupt it by talking). Built around a streaming,
overlapping pipeline — no pause-to-think, no push-to-talk.

Design references: OpenAI Realtime API and ElevenLabs Conversational AI.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## How it differs from ME 3

- **Streaming LLM** — tokens stream in; the first sentence reaches TTS the
  moment it's complete.
- **Streaming TTS** — Piper emits raw PCM (`--output-raw`); the first
  sentence plays while the LLM is still generating the rest.
- **Continuous VAD** — the mic never closes; AEC subtracts the agent's own
  voice from the mic, so the VAD hears only you and stays live while it speaks.
- **Debounced barge-in** — speak for ≥300 ms to interrupt; a cough won't
  kill playback.

## Pi storage note

Nothing big is downloaded. The LLM runs on **Ollama cloud models** by default
(weights stream from Ollama's cloud — nothing lands on the Pi), and ASR stays
on the OpenAI Whisper API (no local Whisper weights). TTS reuses your existing
Piper install. Echo cancellation uses SpeexDSP if `libspeexdsp` is present
(setup.sh installs it, ~10–20 MB), otherwise a pure-NumPy NLMS fallback — so
there's no hard dependency. If your Piper is < 1.2 (no `--output-raw`), it
falls back to sentence-granularity WAV streaming.

## Run

From the `ME 3/` directory (parent of this folder):

```bash
source .venv/bin/activate

# LLM — Ollama cloud models (default; nothing is downloaded to the Pi)
export OLLAMA_API_KEY=ollama_...        # create one at ollama.com
export OLLAMA_MODEL=gemma4:31b          # any model from ollama.com/api/tags

# ASR — still OpenAI Whisper (needs its own key)
export OPENAI_API_KEY=sk-...

export PIPER_BINARY=$HOME/piper/piper
export PIPER_MODEL=$HOME/piper/en_US-lessac-medium.onnx
export PIPER_CONFIG=$HOME/piper/en_US-lessac-medium.onnx.json
export PIPER_LD_LIBRARY_PATH=$HOME/piper

python ME3-FullDuplex/agent_orchestrator.py
```

### LLM backend

The LLM is pluggable via `LLM_BACKEND` (defaults to `ollama-cloud`). All
three backends are OpenAI-compatible, so the streaming code is identical:

| `LLM_BACKEND` | `base_url` | key | model env | Notes |
|---|---|---|---|---|
| `ollama-cloud` *(default)* | `https://ollama.com/v1` | `OLLAMA_API_KEY` | `OLLAMA_MODEL` | Cloud models — **no weights on the Pi** |
| `ollama-local` | `http://localhost:11434/v1/` | (ignored) | `OLLAMA_MODEL` | Local Ollama server; model uses Pi storage |
| `openai` | `https://api.openai.com/v1` | `OPENAI_API_KEY` | `LLM_MODEL` | Original ME 3 default |

> **Why two keys?** The LLM (Ollama cloud) and the ASR (OpenAI Whisper) are
> separate services. `OLLAMA_API_KEY` authenticates the LLM; `OPENAI_API_KEY`
> is still needed for Whisper. If you later move ASR to a local Whisper
> model, you can drop `OPENAI_API_KEY` entirely.

Optional tuning (env vars):

| Var | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `ollama-cloud` | `ollama-cloud` / `ollama-local` / `openai` |
| `OLLAMA_MODEL` | `gemma4:31b` | Model name (Ollama backends) |
| `LLM_MODEL` | `gpt-4o-mini` | Model name (OpenAI backend) |
| `BARGEIN_DEBOUNCE_MS` | `300` | How long you must speak to interrupt |
| `AEC_DELAY_MS` | `40` | Speaker→mic latency applied to the AEC reference |
| `AEC_FILTER_TAIL_S` | `0.2` | AEC adaptive-filter length (seconds) |
| `TELEMETRY_FILE` | `./telemetry.jsonl` | Where latency/interaction data is logged |

## Telemetry

Same dashboard and benchmark as ME 3 work with this version — they read
the same JSONL log. New category: `tts_first_chunk` (first sentence text →
first PCM frame played).

```bash
streamlit run dashboard.py        # from ME 3/
python latency_benchmark.py       # from ME 3/
```

## Files

```
ME3-FullDuplex/
├── ARCHITECTURE.md        ← design
├── README.md              ← this file
├── setup.sh               ← extends parent setup (no new big deps)
├── __init__.py
├── streaming_llm.py       ← streaming LLM + sentence splitter
├── streaming_tts.py       ← streaming Piper + playback + AEC reference feed
├── aec.py                 ← acoustic echo cancellation (SpeexDSP / NumPy NLMS)
├── agent_orchestrator.py  ← state machine, overlapping pipeline
└── vad_system/
    ├── __init__.py
    └── vad_manager.py     ← ME 3 VAD + AEC + barge-in debounce
```
