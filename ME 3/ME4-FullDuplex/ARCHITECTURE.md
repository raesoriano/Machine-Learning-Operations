# ME4-FullDuplex — Architecture

## What changed from ME 3

| ME 3 (current) | ME4-FullDuplex |
|---|---|
| Half-duplex: VAD only meaningful when agent is silent | True full-duplex: VAD runs continuously, even while agent is speaking |
| No echo cancellation — agent hears itself | SpeexDSP AEC subtracts agent's own voice from mic input |
| TTS is blocking: synthesize full file → play → wait | Streaming TTS: Piper outputs WAV in chunks, played as they arrive |
| Barge-in = hard stop: any VAD trigger kills TTS immediately | Graceful barge-in: user must speak ≥300 ms before TTS stops (debounce) |
| LLM is a single blocking call | Streaming LLM: tokens arrive as they're generated, first chunk fed to TTS immediately |
| One-shot pipeline: VAD → ASR → LLM → TTS → play | Overlapping pipeline: next ASR can start while TTS is still playing |

## Design principles (from OpenAI Realtime API / ElevenLabs)

1. **The agent never "stops to think."** While the LLM is generating, the
   mic stays open. While TTS is playing, the mic stays open. The only thing
   that changes is *what we do with* the mic input (AEC'd or not).

2. **Echo cancellation is non-negotiable.** Without AEC, the agent's own
   voice triggers the VAD and causes false interruptions. SpeexDSP's
   `speex_echo_cancellation()` runs in the audio thread, in real time,
   on every 512-sample frame. Latency: <1 ms per frame on a Pi 4.

3. **Streaming everywhere.** LLM tokens stream → sentence boundary detected
   → first sentence fed to Piper → Piper's WAV output streamed to the
   speaker. The user hears the first word ~200–400 ms after the LLM starts
   generating, not after the full response is synthesized.

4. **Barge-in is a policy, not a reflex.** A single VAD trigger (could be
   a cough, a dog, the agent's own echo that slipped through AEC) should
   not kill the agent mid-sentence. We require ≥300 ms of continuous
   post-AEC speech before interrupting. This is the same heuristic OpenAI
   uses in their Realtime API.

5. **The pipeline is a state machine, not a linear chain.** The agent can
   be in any of these states, and transitions can happen at any time:

```
                    ┌─────────────────────────────────────────────┐
                    │                                             │
                    ▼                                             │
            ┌─────────────┐    speech end    ┌──────────────┐     │
            │  LISTENING  │─────────────────▶│  THINKING    │     │
            │  (mic open, │                  │  (LLM gen,   │     │
            │   AEC idle) │                  │   mic open,  │     │
            └─────────────┘                  │   AEC idle)  │     │
                    ▲                        └──────────────┘     │
                    │                               │             │
                    │            first token        │             │
                    │◀──────────────────────────────┘             │
                    │                                             │
                    │    speech start (≥300ms)                    │
                    │◀────────────────────────────────────────────┘
                    │
                    │    TTS playing + user speaks ≥300ms
                    │
            ┌─────────────┐
            │  SPEAKING   │────────────────────────────────────────┘
            │  (TTS out,  │
            │   mic open, │
            │   AEC on)   │
            └─────────────┘
```

## Component breakdown

### 1. `aec.py` — Acoustic Echo Cancellation (NEW)

```
mic_in (16kHz int16, 512 samples)
        │
        ▼
┌─────────────────────┐
│  SpeexDSP AEC       │  ← reference: what we're playing to the speaker
│  speex_echo_cancel()│
└─────────────────────┘
        │
        ▼
  clean_audio (16kHz int16, 512 samples)  →  VAD
```

- Uses `speexdsp` Python bindings (or C extension via `ctypes`).
- Maintains a **reference ring buffer** of the last ~200 ms of TTS output
  (the "far-end" signal). SpeexDSP needs this to know what to subtract.
- The reference buffer is fed by the TTS playback thread every frame.
- AEC tail length: 256 samples (16 ms) — enough for typical Pi speaker
  latency. Configurable via `AEC_TAIL_MS` env var.

**Fallback:** If `speexdsp` isn't available, the system degrades gracefully
to ME 3 behavior (no AEC, but still full-duplex with the 300 ms barge-in
debounce). This keeps the Pi setup simple.

### 2. `streaming_tts.py` — Streaming Piper TTS (NEW)

ME 3's `synthesize_bytes()` runs Piper as a subprocess and waits for the
full WAV. ME4 needs to start playing *before* synthesis is complete.

```
text (streaming from LLM)
        │
        ▼ sentence boundary detector
   "Hello, how are you today?"  ← first sentence
        │
        ▼
   Piper subprocess (stdin pipe)
        │
        ▼ WAV chunks (as they're written)
   ┌──────────────────────────────────┐
   │  Playback thread                │
   │  - reads WAV from pipe/file     │
   │  - resamples to 16kHz if needed │
   │  - writes to PyAudio output     │
   │  - feeds AEC reference buffer   │
   └──────────────────────────────────┘
```

- Piper is invoked with `--output-raw` (raw PCM, no WAV header) so we can
  stream the bytes directly without waiting for the file to close.
- A **sentence boundary detector** splits the LLM stream on `.`, `!`, `?`,
  `\n`, or after 200 characters (whichever comes first). This keeps Piper
  calls short (~1–2 s of audio each) so the first sentence starts playing
  quickly.
- The playback thread runs in its own thread, independent of the event loop.
  It writes to the PyAudio output stream and simultaneously pushes each
  frame into the AEC reference buffer.

### 3. `streaming_llm.py` — Streaming LLM (NEW)

ME 3 calls `client.chat.completions.create()` and waits for the full
response. ME4 uses `stream=True` and yields tokens as they arrive.

```
user transcript
        │
        ▼
   OpenAI chat.completions.create(stream=True)
        │
        ▼ token by token
   "I" → " think" → " the" → " answer" → " is" → " 42."
        │
        ▼ sentence boundary detector
   "I think the answer is 42."  →  streaming_tts
```

- Uses the same OpenAI client as ME 3 (no new dependency).
- The stream is consumed in an `async for` loop on the event loop.
- Each sentence is immediately handed to `streaming_tts` for playback.
- While the LLM is still generating sentence 2, sentence 1 is already
  being spoken. This is the key to "the agent never stops to think."

### 4. `vad_system/vad_manager.py` — Modified VAD

Changes from ME 3:
- **Accepts pre-processed audio** (post-AEC) instead of raw mic. The AEC
  module feeds it, not PyAudio directly. This is a one-line change: the
  VAD's audio source becomes a `queue.Queue` that AEC pushes into.
- **300 ms barge-in debounce.** `on_speech_start` is only fired after the
  VAD has detected continuous speech for ≥300 ms (3 consecutive 512-sample
  frames at 16 kHz ≈ 96 ms each, so ~3–4 frames). A single-frame blip
  (echo, cough, noise) does not trigger barge-in.
- **`is_agent_speaking` flag.** When the agent is speaking, the VAD
  threshold is raised slightly (0.6 vs 0.5) to be more conservative about
  triggering barge-in. This reduces false interruptions from residual echo
  that AEC didn't fully cancel.

### 5. `agent_orchestrator.py` — Rewritten state machine

The ME 3 orchestrator is a linear async function. ME4 is a proper state
machine with concurrent tasks:

```python
class FullDuplexAgent:
    # State
    state: State  # LISTENING, THINKING, SPEAKING

    # Audio
    mic_stream: pyaudio.Stream       # input
    spk_stream: pyaudio.Stream       # output
    aec: AEC                         # echo cancellation
    vad: VADManager                  # voice detection (post-AEC)

    # Pipeline
    llm: StreamingLLM
    tts: StreamingTTS
    playback_thread: Thread          # plays TTS audio + feeds AEC ref

    # Barge-in
    _barge_in_timer: Optional[float]  # timestamp of first VAD trigger
    BARGE_IN_THRESHOLD_MS = 300

    async def run(self):
        # Run all concurrent tasks
        await asyncio.gather(
            self._audio_loop(),      # mic → AEC → VAD
            self._state_machine(),   # handles state transitions
            self._playback_loop(),   # TTS output → speaker + AEC ref
        )
```

**Key methods:**

- `_audio_loop()`: Reads mic frames, runs AEC, feeds VAD. Runs in a thread
  (blocking PyAudio read). VAD callbacks are dispatched to the event loop.

- `_state_machine()`: The brain. Waits on an `asyncio.Event` that's set by
  VAD callbacks. On each event, checks current state and transitions:
  - LISTENING + speech_end → start LLM (THINKING)
  - THINKING + first token → start TTS (SPEAKING)
  - SPEAKING + barge_in (≥300 ms) → stop TTS, start new ASR (LISTENING)
  - SPEAKING + TTS done → LISTENING

- `_playback_loop()`: Reads from TTS output queue, writes to speaker,
  pushes each frame into AEC reference buffer.

### 6. `telemetry.py` — Extended

New telemetry categories:
- `aec_latency` — time per AEC frame (should be <1 ms)
- `tts_first_byte` — time from first LLM token to first TTS audio byte
- `barge_in_rejected` — count of VAD triggers that were <300 ms (debounced)
- `barge_in_accepted` — count of VAD triggers that caused actual interruption
- `overlap_ms` — how long user speech and agent speech overlapped before
  barge-in was accepted (measures responsiveness)

### 7. `dashboard.py` — Extended

New panels:
- **AEC status**: on/off, tail length, residual echo level (RMS of AEC
  output during agent speech — should be near silence)
- **Barge-in stats**: accepted vs rejected ratio, average overlap time
- **Pipeline overlap**: a timeline showing when LLM generation, TTS
  synthesis, and TTS playback were active (should overlap significantly)

## File structure

```
ME 3/
├── (existing ME 3 files — unchanged)
│   ├── agent_orchestrator.py
│   ├── asr_server/
│   ├── tts_server/
│   ├── vad_system/
│   ├── telemetry.py
│   ├── dashboard.py
│   ├── latency_benchmark.py
│   └── setup.sh
│
└── ME4-FullDuplex/          ← NEW
    ├── ARCHITECTURE.md      ← this file
    ├── README.md
    ├── setup.sh             ← extends ME 3 setup with speexdsp
    ├── aec.py               ← NEW: SpeexDSP echo cancellation
    ├── streaming_tts.py     ← NEW: streaming Piper TTS
    ├── streaming_llm.py     ← NEW: streaming LLM with sentence splitting
    ├── agent_orchestrator.py← REWRITTEN: state machine, concurrent tasks
    ├── vad_system/
    │   └── vad_manager.py   ← MODIFIED: post-AEC input, barge-in debounce
    ├── telemetry.py         ← EXTENDED: AEC, barge-in, overlap metrics
    ├── dashboard.py         ← EXTENDED: AEC panel, barge-in stats
    └── latency_benchmark.py ← EXTENDED: new latency categories
```

## Dependencies (new)

| Package | Purpose | Pi install |
|---|---|---|
| `speexdsp` (or `python-speexdsp`) | AEC | `pip install speexdsp` (needs `libspeexdsp-dev`) |
| *(no new Python deps for LLM/TTS — reuses OpenAI + Piper)* | | |

System package: `sudo apt install libspeexdsp-dev`

**If speexdsp is unavailable:** the system falls back to no-AEC mode.
Barge-in still works (300 ms debounce), but the agent may occasionally
interrupt itself. This is acceptable for a first pass — AEC is the
"nice to have" that makes it feel like a phone call.

## Latency budget (target)

| Stage | ME 3 (measured) | ME4 target |
|---|---|---|
| VAD trigger → speech end | ~200 ms | ~200 ms (same) |
| ASR turnaround | ~800 ms (Whisper API) | ~800 ms (same, or ~200 ms with local Whisper) |
| LLM first token | ~300 ms | ~300 ms (same) |
| TTS first byte | ~500 ms (full synth) | **~150 ms** (streaming, first sentence) |
| TTS first audio out | ~1000 ms total | **~400 ms** (streaming) |
| **Total voice-to-voice** | **~2300 ms** | **~1800 ms** |
| Barge-in response | instant (but unreliable) | **~300 ms** (debounced, reliable) |

The biggest win is TTS first byte: going from "synthesize the whole
response then play" to "synthesize sentence 1, play it, synthesize
sentence 2 in parallel" cuts perceived latency by ~500–600 ms.

## What we're NOT doing (yet)

- **Local Whisper ASR** — would cut ASR latency from 800 ms to ~200 ms,
  but adds a 150 MB model download and CPU load. Future iteration.
- **AEC training/adaptation** — SpeexDSP's AEC is fixed-tap. WebRTC AEC3
  would be better but is much harder to integrate. Future iteration.
- **Full-duplex LLM** (agent can "hear" itself and react) — not needed
  for a voice assistant. The AEC handles the echo; the LLM doesn't need
  to know the agent is speaking.
- **Multi-speaker diarization** — out of scope.

## Build order

1. `aec.py` — the foundation. Everything else depends on clean audio.
2. `streaming_tts.py` — Piper streaming + playback thread.
3. `streaming_llm.py` — OpenAI streaming + sentence splitter.
4. `vad_system/vad_manager.py` — modify for post-AEC input + debounce.
5. `agent_orchestrator.py` — the state machine that ties it all together.
6. `telemetry.py` + `dashboard.py` + `latency_benchmark.py` — observability.
7. `setup.sh` + `README.md` — deployment.
