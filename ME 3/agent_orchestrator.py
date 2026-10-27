import asyncio
import os
import struct
import tempfile
import time

import numpy as np

from vad_system.vad_manager import VADManager
from asr_server.transcribe import transcribe_bytes
from tts_server.synthesize import synthesize_bytes
from telemetry import default as telemetry


def pcm_bytes_to_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    """
    Wrap raw 16-bit mono PCM bytes in a minimal WAV container.

    Whisper (and most ASR backends) expect a proper audio file, not bare PCM.
    """
    num_frames = len(pcm) // 2
    data_size = len(pcm)
    # RIFF header
    header = b"RIFF" + struct.pack("<I", 36 + data_size) + b"WAVE"
    # fmt chunk (PCM, mono, 16-bit)
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate,
                                    sample_rate * 2, 2, 16)
    # data chunk
    header += b"data" + struct.pack("<I", data_size)
    return header + pcm


class FullDuplexAgent:
    def __init__(self):
        self.vad = VADManager()
        self.is_speaking = False
        self.stop_tts = False

        # Timing anchors for telemetry (monotonic seconds).
        self._speech_start_ts = None   # when VAD fired speech-start
        self._turn = 0

        # Callbacks for VAD (async — dispatched onto the event loop by VADManager)
        self.vad.on_speech_start = self.handle_speech_start
        self.vad.on_speech_end = self.handle_speech_end

    async def handle_speech_start(self):
        print("[Agent] User started speaking... Interrupting TTS.")
        now = time.monotonic()
        self._speech_start_ts = now
        self.stop_tts = True
        if self.is_speaking:
            # In a real implementation, this would kill the audio playback
            # process/stream immediately.
            self.is_speaking = False
            # Interruption latency: from the moment the user started speaking
            # to the moment we acted on it (stopped TTS). In this simulated
            # playback the action is immediate, so it's ~0; with a real player
            # this would capture the kill/stream-stop delay.
            telemetry.record_latency("interruption", now - now, turn=self._turn)

    async def handle_speech_end(self):
        print("[Agent] User stopped speaking. Processing request...")
        self._turn = telemetry.next_turn()
        turn = self._turn

        # VAD trigger latency: speech-start -> speech-end (utterance length as a
        # proxy for detection-to-finalize; real VAD trigger is measured in the
        # audio thread). Recorded here so every completed turn has a sample.
        if self._speech_start_ts is not None:
            telemetry.record_latency(
                "vad_trigger", time.monotonic() - self._speech_start_ts, turn=turn
            )
        self._speech_start_ts = None

        # 1. Capture the audio buffer from VAD (raw int16 PCM)
        pcm = self.vad.get_last_utterance()
        if not pcm:
            return

        # 2. ASR — wrap PCM in a WAV container, then transcribe the bytes
        wav = pcm_bytes_to_wav(pcm, self.vad.sample_rate)
        asr_start = time.monotonic()
        try:
            text = await asyncio.get_running_loop().run_in_executor(
                None, transcribe_bytes, wav
            )
        except Exception as e:
            print(f"[Agent] ASR error: {e}")
            return
        asr_ms = telemetry.record_latency(
            "asr_turnaround", time.monotonic() - asr_start, turn=turn
        )
        print(f"[User]: {text}")
        telemetry.record_interaction("User", text, latency_ms=asr_ms, turn=turn)
        if not text.strip():
            return

        # 3. LLM Processing
        llm_start = time.monotonic()
        response_text = await self.call_llm(text)
        llm_ms = telemetry.record_latency(
            "llm_ttfb", time.monotonic() - llm_start, turn=turn
        )
        print(f"[AI]: {response_text}")
        telemetry.record_interaction("OnIt", response_text,
                                     latency_ms=llm_ms, turn=turn)

        # 4. TTS
        await self.speak(response_text, turn=turn)

    async def call_llm(self, text):
        # Simulated LLM call (swap for a real provider in production)
        await asyncio.sleep(0.5)
        return f"I heard you say '{text}'. I am your full-duplex agent."

    async def speak(self, text, turn: int | None = None):
        self.stop_tts = False
        self.is_speaking = True

        # Synthesize with local Piper TTS (no API key). synthesize_bytes
        # returns (path, ok).
        tts_start = time.monotonic()
        audio_path, ok = await asyncio.get_running_loop().run_in_executor(
            None, synthesize_bytes, text
        )
        if ok:
            telemetry.record_latency(
                "tts_synthesis", time.monotonic() - tts_start, turn=turn
            )
        if not ok:
            print("[Agent] TTS synthesis failed — check Piper setup.")
            self.is_speaking = False
            return

        print("[Agent] Playing response...")
        # Simulated playback that can be interrupted. In production this would
        # use a non-blocking audio player (e.g. pygame / simpleaudio) that
        # checks self.stop_tts in a loop.
        playback_duration = 3.0
        start_time = time.time()
        while time.time() - start_time < playback_duration:
            if self.stop_tts:
                print("[Agent] TTS Interrupted!")
                break
            await asyncio.sleep(0.1)

        # Clean up the temp wav.
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except OSError:
            pass

        self.is_speaking = False

    async def run(self):
        print("Starting Full-Duplex Agent... Press Ctrl+C to stop.")
        # Start VAD in a separate thread/async task
        vad_task = asyncio.create_task(self.vad.listen_loop())

        try:
            while True:
                await asyncio.sleep(1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            vad_task.cancel()
            self.vad.stop()


if __name__ == "__main__":
    agent = FullDuplexAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass
