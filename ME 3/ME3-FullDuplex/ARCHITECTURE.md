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
| Agent hears itself (no echo handling) | Echo gate: VAD muted while TTS is playing (+ tail) |
| One-shot pipeline: VAD → ASR → LLM → TTS → play | Overlapping pipeline: next ASR can start while TTS still plays |

## Pi storage budget (16 GB)

The Pi has only 16 GB, so **nothing new big is downloaded**:

| Component | Decision | Why |
|---|---|---|
| ASR | Keep OpenAI Whisper **API** | Local Whisper small/base = 150 MB+ weights; API costs 0 storage |
| Echo cancellation | **Echo gate** (VAD muted during TTS) instead of SpeexDSP AEC | `libspeexdsp-dev` + build toolchain; gate is 0 bytes and good enough for a single speaker + mic |
| TTS | Reuse the **existing Piper** install | No new model download. If `--output-raw` is missing (Piper < 1.2), fall back to WAV-per-sentence streaming |
| VAD | Reuse existing Silero VAD (torch.hub, cached) | Already on the Pi from ME 3 |

Net new downloads: **zero**. Net new system packages: **zero**.

## The pipeline

```
 mic ──► VAD (always on) ──► echo gate ──┬─► utterance buffer ──► ASR (API)
                                         │                              │
 speaker ◄── playback thread ◄── TTS (Piper, streaming) ◄── sentence splitter ◄── LLM (streaming)
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
2. **Listen while speaking.** The mic never closes. The echo gate prevents
   the agent from triggering on its own voice; the barge-in debounce
   prevents a cough or a "hmm" from killing playback.
3. **Graceful barge-in.** If the user speaks for ≥300 ms while the agent is
   talking, TTS is stopped and the agent immediately starts transcribing the
   interruption.
4. **Overlap.** While the agent is still speaking the tail of a reply, a new
   utterance is already being captured and can be transcribed the moment it
   ends — the ASR request is issued in parallel with the remaining playback.

## Echo gate (instead of AEC)

True full-duplex on a phone uses acoustic echo cancellation (AEC) so the mic
hears only the far end. AEC needs a reference signal and a DSP library
(SpeexDSP / WebRTC AEC3) — a build dependency we don't want on a 16 GB Pi.

The echo gate is the cheap, robust approximation:

- While TTS is playing (and for a 250 ms tail after it stops), the VAD's
  speech probability is **clamped to 0**. The agent cannot hear itself.
- The moment playback stops, the VAD is live again.
- Cost: the agent is deaf for exactly as long as it is talking. For a
  single-user, single-speaker setup this is the same trade-off a real
  half-duplex phone makes, and it's what makes barge-in work without AEC:
  if you interrupt, your voice is the *only* thing that can re-open the gate
  (the gate re-arms on the 250 ms tail, and the debounce confirms you're
  really talking).

Limitation, stated plainly: if the user keeps talking *while* the agent is
still talking, the gate stays closed until the agent finishes or the
debounce fires. With the 300 ms debounce and 250 ms tail, an interruption is
recognized within ~550 ms of the user starting to talk.

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
- The playback thread also drives the **echo gate** and reports
  `is_playing()` to the orchestrator.
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
    ├── streaming_tts.py       ← streaming Piper + playback + echo gate
    ├── agent_orchestrator.py  ← state machine, overlapping pipeline
    └── vad_system/
        └── vad_manager.py     ← ME 3 VAD + echo gate + barge-in debounce
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
| `ECHO_GATE_TAIL_MS` | Gate tail after TTS stops | `250` |
