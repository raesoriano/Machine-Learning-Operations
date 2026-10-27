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
- **Continuous VAD** — the mic never closes; an echo gate keeps the agent
  deaf to its own voice while it's speaking.
- **Debounced barge-in** — speak for ≥300 ms to interrupt; a cough won't
  kill playback.

## Pi storage note

Nothing new is downloaded. ASR stays on the OpenAI Whisper API (no local
Whisper weights), echo handling is a zero-dependency gate (no SpeexDSP),
and TTS reuses your existing Piper install. If your Piper is < 1.2 (no
`--output-raw`), it falls back to sentence-granularity WAV streaming.

## Run

From the `ME 3/` directory (parent of this folder):

```bash
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
export PIPER_BINARY=$HOME/piper/piper
export PIPER_MODEL=$HOME/piper/en_US-lessac-medium.onnx
export PIPER_CONFIG=$HOME/piper/en_US-lessac-medium.onnx.json
export PIPER_LD_LIBRARY_PATH=$HOME/piper

python ME3-FullDuplex/agent_orchestrator.py
```

Optional tuning (env vars):

| Var | Default | Meaning |
|---|---|---|
| `BARGEIN_DEBOUNCE_MS` | `300` | How long you must speak to interrupt |
| `ECHO_GATE_TAIL_MS` | `250` | How long the agent stays deaf after it stops talking |
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
├── streaming_tts.py       ← streaming Piper + playback + echo gate
├── agent_orchestrator.py  ← state machine, overlapping pipeline
└── vad_system/
    ├── __init__.py
    └── vad_manager.py     ← ME 3 VAD + echo gate + barge-in debounce
```
