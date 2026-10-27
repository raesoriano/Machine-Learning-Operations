# ME3-FullDuplex — Architecture

A new version of ME 3 (not a new milestone): the same voice agent, rebuilt
around a **continuous, overlapping pipeline** so it behaves like a phone call
instead of a push-to-talk device. Design references: OpenAI Realtime API
(streaming, barge-in, no pause-to-think) and ElevenLabs Conversational AI
(listen while speaking, graceful interruption).

## What changed from ME 3

| ME 3 (current) | ME3-FullDuplex |
|---|---|
| Half-duplex: VAD only meaningful when agent is silent | Continuous VAD: keeps listening while agent speaks |
| TTS is blocking: synthesize full WAV → play → wait | Streaming TTS: Piper emits raw PCM, played as it arrives |
| LLM is one blocking call | Streaming LLM: tokens arrive as generated, first sentence fed to TTS immediately |
| Barge-in = hard stop: any VAD trigger kills TTS | Debounced barge-in: user must speak ≥300 ms before TTS stops |
| Agent hears itself (no echo handling) | **AEC**: the TTS signal is subtracted from the mic, so the VAD hears only the user |
| One-shot pipeline: VAD → ASR → LLM → TTS → play | Overlapping pipeline: next ASR can start while TTS still plays |

## Pi storage budget (16 GB)

The Pi has only 16 GB, so **nothing new big is downloaded**:

| Component | Decision | Why |
|---|---|---|
| ASR | Keep OpenAI Whisper **API** | Local Whisper small/base = 150 MB+ weights; API costs 0 storage |
| Echo cancellation | **AEC** — SpeexDSP if the system lib works, else a pure-NumPy NLMS fallback | `libspeexdsp-dev` (~10–20 MB) for the best suppression; the NumPy fallback needs no extra install |
| TTS | Reuse the **existing Piper** install | No new model download. If `--output-raw` is missing (Piper < 1.2), fall back to WAV-per-sentence streaming |
| VAD | Reuse existing Silero VAD (torch.hub, cached) | Already on the Pi from ME 3 |

Net new downloads: **zero**. Net new system packages: **zero**.

## The pipeline

```
 mic ──▶ AEC ──▶ VAD (always on) ──┬──▶ utterance buffer ──▶ ASR (API)
              ▲                    └──▶ (live while TTS plays)
              └──────── reference (TTS frames) ────────┘
 speaker ◀── playback thread ◀── TTS (Piper, streaming) ◀── sentence splitter ◀── LLM (streaming)
```

State machine (one state at a time, transitions can overlap):

```
        user starts speaking              user stops speaking
 IDLE ─────────────────────────► LISTENING ─────────────────────────► THINKING
  ▲                                                                           │
  │                                                                           │ first sentence out
  │ agent done speaking                                                       ▼
  └──────────────────────────────────────────────────────────────────── SPEAKING
                                                                            │
        user speaks ≥300ms (debounced barge-in)                             │
        ────────────────────────────────────────────────────────────────────┘
                                  (TTS stops, back to LISTENING)
```

Key properties:

1. **No pause-to-think.** The LLM call starts the moment the utterance ends;
   the first TTS chunk starts the moment the first sentence boundary is
   detected in the token stream. No stage waits for the previous stage to
   fully finish.
2. **Listen while speaking.** The mic never closes. The AEC subtracts the
   agent's own TTS from the mic, so the VAD hears only the user and stays
   live the whole time; the barge-in debounce prevents a cough or a "hmm"
   from killing playback.
3. **Graceful barge-in.** If the user speaks for ≥300 ms while the agent is
   talking, TTS is stopped and the agent immediately starts transcribing the
   interruption.
4. **Overlap.** While the agent is still speaking the tail of a reply, a new
   utterance is already being captured and can be transcribed the moment it
   ends — the ASR request is issued in parallel with the remaining playback.

## Acoustic echo cancellation (AEC)

True full-duplex needs the mic to hear only the user, not the agent's own
voice. `aec.py` does this by feeding the exact TTS signal sent to the speaker
into an adaptive filter, which subtracts it from the live mic stream before
the VAD runs.

Two backends, chosen at runtime:

- **SpeexDSP** (preferred) — a real adaptive AEC via ctypes, using the system
  `libspeexdsp` (apt: `libspeexdsp-dev`, ~10–20 MB). It is only used if a
  self-test at init confirms the build actually produces output (some aarch64
  builds export the symbols but are no-ops).
- **Pure-NumPy NLMS** (fallback) — a sample-by-sample NLMS in NumPy, zero
  system dependencies. It converges more slowly than SpeexDSP, so for the best
  suppression on a Pi install `libspeexdsp` (setup.sh does this). The fallback
  guarantees the agent still runs — and still listens while it talks — even
  without the library.

The reference signal is kept at the native TTS rate (22050 Hz) and delayed by
~40 ms (speaker→mic latency) so it lines up with the mic capture. The VAD runs
on the cleaned signal, so it is never deafened while the agent speaks — which
is what makes barge-in real.

## Streaming TTS

`streaming_tts.py`:

- Piper is invoked with `--output-raw` (raw 16-bit PCM, no WAV header) so
  bytes can be read from stdout as they are produced.
- A **sentence splitter** buffers LLM text and emits a sentence when it sees
  `.`, `!`, `?` (or a newline / 200-char overflow). Each sentence is one
  Piper invocation, so the first sentence starts playing while the LLM is
  still generating the rest.
- A **playback thread** reads PCM from the Piper pipe and writes it to the
  PyAudio output stream in 30 ms frames.
- The playback thread feeds **every frame it writes** to the speaker into the
  VAD's AEC (`vad.feed_reference`) and reports `is_playing()` to the
  orchestrator.
- **Fallback:** if the Piper binary doesn't support `--output-raw` (Piper
  < 1.2), each sentence is synthesized to a temp WAV and played whole.
  Streaming still works at sentence granularity — only intra-sentence
  streaming is lost.

## Streaming LLM

`streaming_llm.py`:

- `openai.chat.completions.create(stream=True)`.
- A generator yields **sentences**, not tokens: it accumulates tokens and
  splits on sentence boundaries (same splitter as the TTS side).
- The first sentence is available ~1–2 tokens after the first token arrives,
  which is what removes the "pause to think" feel.

## Barge-in

- VAD `on_speech_start` fires while `SPEAKING` → start a 300 ms timer.
- If speech is still going after 300 ms → confirmed barge-in:
  - stop the playback thread (flush remaining frames),
  - close the Piper subprocess,
  - record `interruption` latency (speech-start → TTS-stopped),
  - transition to `LISTENING` with the current utterance buffer.
- If speech ends before 300 ms → false trigger (noise/cough), ignore.

## Files

```
ME 3/
├── (existing ME 3 files — unchanged)
│
└── ME3-FullDuplex/            ← NEW (a version of ME 3)
    ├── ARCHITECTURE.md        ← this file
    ├── README.md              ← how to run
    ├── setup.sh               ← extends parent setup (no new big deps)
    ├── __init__.py
    ├── streaming_llm.py       ← streaming LLM + sentence splitter
    ├── streaming_tts.py       ← streaming Piper + playback + AEC reference feed
    ├── aec.py                 ← acoustic echo cancellation (SpeexDSP / NumPy NLMS)
    ├── agent_orchestrator.py  ← state machine, overlapping pipeline
    └── vad_system/
        └── vad_manager.py     ← ME 3 VAD + AEC + barge-in debounce
```

Reused from the parent `ME 3/` (via `sys.path`): `asr_server/transcribe.py`
(Whisper API), `telemetry.py` (latency + interaction log),
`tts_server/synthesize.py` (Piper env-var config: `PIPER_BINARY`,
`PIPER_MODEL`, `PIPER_CONFIG`, `PIPER_LD_LIBRARY_PATH`).

## Telemetry

Same `telemetry.py` as ME 3, same categories, plus:
- `llm_ttfb` — speech-end → first LLM token
- `tts_first_chunk` — first sentence text → first PCM byte played
- `interruption` — user speech-start → TTS stopped (only on confirmed barge-in)

## Env vars

| Var | Meaning | Default |
|---|---|---|
| `OPENAI_API_KEY` | LLM + Whisper API | required |
| `PIPER_BINARY` | Piper executable | `~/piper/piper` |
| `PIPER_MODEL` | Piper voice model (.onnx) | `~/piper/en_US-lessac-medium.onnx` |
| `PIPER_CONFIG` | Piper model config (.json) | `~/piper/en_US-lessac-medium.onnx.json` |
| `PIPER_LD_LIBRARY_PATH` | LD_LIBRARY_PATH for Piper | `~/piper` |
| `TELEMETRY_FILE` | JSONL telemetry log | `./telemetry.jsonl` |
| `BARGEIN_DEBOUNCE_MS` | Barge-in debounce | `300` |
| `AEC_DELAY_MS` | Speaker→mic latency applied to the AEC reference | `40` |
| `AEC_FILTER_TAIL_S` | AEC adaptive-filter length (seconds) | `0.2` |
| `AEC_MU` | NLMS step size (fallback backend) | `0.05` |
