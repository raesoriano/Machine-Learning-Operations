"""
ME3-FullDuplex orchestrator.

A state machine that runs the overlapping voice pipeline:

    mic → AEC → VAD (always on) → ASR (Whisper API)
        → LLM (streaming) → TTS (streaming Piper) → speaker
                    ↑________________ reference ______________|

The TTS playback feeds every frame it writes to the speaker into the AEC,
which subtracts the agent's own voice from the mic before the VAD runs. The
VAD is therefore live the whole time — including while the agent is talking —
so it genuinely listens while it speaks.

States:
    IDLE        — waiting for the user to start speaking
    LISTENING   — user is speaking; capturing the utterance
    THINKING    — utterance captured; ASR + LLM running
    SPEAKING    — TTS playing; VAD still live (AEC-cleaned) for barge-in

The "no pause to think" property comes from streaming: the LLM call starts
the moment the utterance ends, and the first sentence reaches the TTS the
moment the first sentence boundary appears in the token stream.

Barge-in: if the user speaks ≥ debounce_ms while SPEAKING, TTS is stopped
and the agent immediately starts on the interruption.

Concurrency note: VAD callbacks are scheduled onto the event loop from the
audio thread, so they can interleave with an in-flight pipeline task at any
await point. A generation counter (`_gen`) invalidates stale pipeline runs:
when a new utterance or a barge-in happens, the old task sees its generation
no longer matches and exits without touching shared state.
"""

import asyncio
import os
import struct
import sys
import time
from typing import Optional

# Make the parent ME 3 directory importable so we reuse its ASR helper,
# telemetry, and Piper config without duplicating them.
#
# IMPORTANT: the parent goes LAST on sys.path. This package ships its own
# vad_system/ (with the AEC + barge-in debounce), and it must win
# over the parent's plain vad_system/. The parent is only there to supply
# asr_server/ and telemetry, which this package does not have.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
if _PARENT not in sys.path:
    sys.path.append(_PARENT)

from vad_system.vad_manager import VADManager          # noqa: E402  (ME3-FullDuplex copy)
from asr_server.transcribe import transcribe_bytes     # noqa: E402  (ME 3, Whisper API)
from streaming_llm import stream_sentences             # noqa: E402
from streaming_tts import StreamingTTS                 # noqa: E402
from telemetry import default as telemetry             # noqa: E402  (ME 3)

BARGEIN_DEBOUNCE_MS = int(os.environ.get("BARGEIN_DEBOUNCE_MS", "300"))


def pcm_bytes_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a minimal WAV container (for ASR)."""
    data_size = len(pcm)
    header = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                    sample_rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", data_size)
    return header + pcm


class FullDuplexAgent:
    """The ME3-FullDuplex voice agent state machine."""

    IDLE = "IDLE"
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"

    def __init__(self):
        self.state = self.IDLE
        self.vad = VADManager(
            on_speech_start=self.on_speech_start,
            on_speech_end=self.on_speech_end,
            debounce_ms=BARGEIN_DEBOUNCE_MS,
        )
        self.tts = StreamingTTS(vad=self.vad)

        # Timing anchors for telemetry (monotonic seconds).
        self._speech_start_ts: Optional[float] = None
        self._speech_end_ts: Optional[float] = None
        self._asr_done_ts: Optional[float] = None
        self._llm_first_token_ts: Optional[float] = None
        self._turn = 0
        self._gen = 0  # pipeline generation; bumped on every new utterance

    # ------------------------------------------------------------------ #
    # VAD callbacks (scheduled onto the event loop from the audio thread)
    # ------------------------------------------------------------------ #
    async def on_speech_start(self):
        now = time.monotonic()
        self._speech_start_ts = now

        if self.state == self.SPEAKING:
            # Barge-in: the VAD debounce has already confirmed the user has
            # been speaking for ≥ debounce_ms. Stop TTS and hand control to
            # the user.
            self.tts.stop()
            telemetry.record_latency(
                "interruption", now - self._speech_start_ts, turn=self._turn
            )
            print("[Agent] Barge-in confirmed — stopping TTS")
            self.state = self.LISTENING
            return

        if self.state == self.IDLE:
            self.state = self.LISTENING
            print("[Agent] User started speaking")

    async def on_speech_end(self):
        if self.state != self.LISTENING:
            # Speech ended while THINKING/SPEAKING (e.g. the user finished an
            # interruption before the agent resumed) — nothing to do.
            return
        self._speech_end_ts = time.monotonic()
        self.state = self.THINKING
        self._gen += 1
        gen = self._gen
        asyncio.create_task(self._handle_utterance(gen))

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    async def _handle_utterance(self, gen: int):
        turn = telemetry.next_turn()
        self._turn = turn
        t0 = self._speech_end_ts or time.monotonic()

        def stale() -> bool:
            return gen != self._gen

        # 1. Capture the utterance (raw int16 PCM at 16 kHz).
        pcm = self.vad.get_last_utterance()
        if not pcm:
            if not stale():
                self.state = self.IDLE
            return
        telemetry.record_latency(
            "vad_trigger", t0 - (self._speech_start_ts or t0), turn=turn
        )
        print("[Agent] User stopped speaking. Processing request...")

        # 2. ASR (Whisper API) — run in executor so the loop stays free.
        loop = asyncio.get_running_loop()
        try:
            text = await loop.run_in_executor(
                None, transcribe_bytes, pcm_bytes_to_wav(pcm)
            )
        except RuntimeError as e:
            print(f"[Agent] ASR failed: {e}")
            if not stale():
                self.state = self.IDLE
            return
        if stale():
            return  # superseded by a newer utterance; drop this one
        asr_ms = (time.monotonic() - t0) * 1000.0
        telemetry.record_latency("asr_turnaround", time.monotonic() - t0, turn=turn)
        telemetry.record_interaction("User", text, asr_ms, turn)
        print(f"[User] {text}")
        if not text or not text.strip():
            self.state = self.IDLE
            return
        self._asr_done_ts = time.monotonic()

        # 3. LLM (streaming) + 4. TTS (streaming) — overlapping.
        self.state = self.SPEAKING
        self.tts.start()

        # Telemetry hook: first PCM frame played.
        def _on_first_frame(idx, ts):
            if self._llm_first_token_ts is not None:
                telemetry.record_latency(
                    "tts_first_chunk", ts - self._llm_first_token_ts, turn=turn
                )

        self.tts.on_first_frame = _on_first_frame

        # Feed sentences to TTS as the LLM produces them.
        first_sentence = True
        try:
            for sentence in stream_sentences(text):
                if stale() or self.state != self.SPEAKING:
                    # Barge-in (or a newer utterance) happened mid-stream.
                    break
                if first_sentence:
                    self._llm_first_token_ts = time.monotonic()
                    telemetry.record_latency(
                        "llm_ttfb",
                        self._llm_first_token_ts - (self._asr_done_ts or t0),
                        turn=turn,
                    )
                    first_sentence = False
                self.tts.enqueue(sentence)
        except RuntimeError as e:
            print(f"[Agent] LLM failed: {e}")

        self.tts.finish()

        # Wait for playback to drain (or until barge-in stops it).
        while self.tts.is_playing() and self.state == self.SPEAKING:
            await asyncio.sleep(0.05)

        if not stale() and self.state == self.SPEAKING:
            # Normal completion.
            self.state = self.IDLE
            print("[Agent] Done speaking")
        # If state is LISTENING, a barge-in already handled the transition.

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #
    async def run(self):
        """Run the agent until Ctrl+C."""
        print("=" * 50)
        print(" ME3-FullDuplex — continuous voice agent")
        print(f" (full-duplex: AEC backend = {self.vad.aec_backend};")
        print("  barge-in with ≥300 ms speech)")
        print("=" * 50)
        try:
            await self.vad.listen_loop()
        except KeyboardInterrupt:
            print("\n[Agent] Shutting down")
        finally:
            self.tts.close()
            self.vad.stop()


if __name__ == "__main__":
    agent = FullDuplexAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass
