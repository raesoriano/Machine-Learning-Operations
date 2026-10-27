"""
VAD Manager for ME3-FullDuplex.

Based on the ME 3 VADManager (Silero VAD), with two additions:

1. Echo gate — while the agent is speaking (and for a short tail after),
   the speech probability is clamped to 0 so the agent cannot trigger on
   its own voice. This replaces full AEC with a zero-dependency
   approximation that is good enough for a single speaker + mic on a Pi.

2. Barge-in debounce — a speech-start event is only reported after the
   user has been speaking for `debounce_ms` (default 300 ms). This
   prevents a cough or a "hmm" from killing TTS playback.

Everything else (chunked capture, utterance buffering, async callback
dispatch) is unchanged from ME 3.
"""

import asyncio
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch
import pyaudio

# Canonical Silero VAD source (was incorrectly 'snyprow/silero-vad' before).
SILERO_REPO = "snakers4/silero-vad"
SILERO_MODEL = "silero_vad"
# Cache the model so repeated instantiations don't re-download it.
SILERO_CACHE = "silero_vad"


class VADManager:
    """
    VAD Manager using Silero VAD for high-accuracy voice detection on
    Raspberry Pi, with an echo gate and barge-in debounce.

    Callbacks (`on_speech_start`, `on_speech_end`) may be either plain
    callables or async coroutines. Async callbacks are scheduled onto the
    running event loop from the audio thread, so they actually execute.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
        debounce_ms: int = 300,
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        # Barge-in debounce: speech must persist this long before
        # on_speech_start fires.
        self.debounce_ms = debounce_ms

        # Model is loaded lazily (see _load_model) so importing/instantiating
        # this class never triggers a network download.
        self.model = None
        self.utils = None

        self.is_speaking = False
        self._utterance_chunks = []

        # Echo gate state. set_echo_gate(True) is called by the TTS
        # playback thread while the agent is speaking; the audio thread
        # clamps speech probability to 0 while the gate is closed.
        self._echo_gate = threading.Event()

        # Barge-in debounce state (audio thread only — no lock needed).
        self._candidate_start_ts: Optional[float] = None

        # The running event loop, captured by listen_loop() so the audio
        # thread can dispatch async callbacks onto it.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def _load_model(self):
        """Load Silero VAD once, caching it in the torch.hub cache dir."""
        if self.model is not None:
            return
        self.model, self.utils = torch.hub.load(
            repo_or_dir=SILERO_REPO,
            model=SILERO_MODEL,
            source="github",
            force_reload=False,
            cache_dir=SILERO_CACHE,
        )

    # ------------------------------------------------------------------ #
    # Echo gate
    # ------------------------------------------------------------------ #
    def set_echo_gate(self, closed: bool):
        """
        Close (True) or open (False) the echo gate.

        While closed, the VAD reports no speech regardless of what the mic
        hears — this is how the agent is kept deaf to its own voice while
        TTS is playing. The TTS playback thread closes the gate before the
        first frame and opens it (after a short tail) when playback ends.
        """
        if closed:
            self._echo_gate.set()
        else:
            self._echo_gate.clear()

    @property
    def echo_gate_closed(self) -> bool:
        return self._echo_gate.is_set()

    # ------------------------------------------------------------------ #
    # Callback dispatch (thread -> event loop)
    # ------------------------------------------------------------------ #
    def _fire(self, callback):
        if callback is None:
            return
        try:
            result = callback()
        except Exception as e:  # never let a callback crash the audio loop
            print(f"[VAD] callback error: {e}")
            return
        # If the callback returned a coroutine, schedule it on the loop.
        if asyncio.iscoroutine(result) and self._loop is not None:
            asyncio.run_coroutine_threadsafe(result, self._loop)

    # ------------------------------------------------------------------ #
    # Audio loop
    # ------------------------------------------------------------------ #
    def _process_chunk(self, audio_int16: np.ndarray) -> bool:
        """Return the speech probability for a single int16 chunk."""
        audio_float32 = audio_int16.astype(np.float32) / 32768.0
        tensor_audio = torch.from_numpy(audio_float32)
        return self.model(tensor_audio, self.sample_rate).item()

    def _audio_thread(self):
        """Blocking audio capture + VAD inference, run in a worker thread."""
        try:
            while True:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)

                # Echo gate: while the agent is speaking, we are deaf.
                if self._echo_gate.is_set():
                    # Drop any in-progress candidate; nothing is speech.
                    self._candidate_start_ts = None
                    if self.is_speaking:
                        # Speech was in progress when the gate closed
                        # (e.g. TTS started mid-utterance). End it quietly.
                        self.is_speaking = False
                    continue

                speech_prob = self._process_chunk(audio_int16)

                if speech_prob > 0.5 and not self.is_speaking:
                    # Candidate start — wait for the debounce to confirm.
                    if self._candidate_start_ts is None:
                        self._candidate_start_ts = time.monotonic()
                        self._utterance_chunks = []
                    elif (time.monotonic() - self._candidate_start_ts) * 1000.0 \
                            >= self.debounce_ms:
                        # Confirmed: real speech start.
                        self.is_speaking = True
                        self._candidate_start_ts = None
                        print("[VAD] Speech Started (debounced)")
                        self._fire(self.on_speech_start)

                elif speech_prob < 0.3:
                    # Either the candidate died (debounce not met) or
                    # confirmed speech ended.
                    if self.is_speaking:
                        self.is_speaking = False
                        self._candidate_start_ts = None
                        print("[VAD] Speech Ended")
                        self._fire(self.on_speech_end)
                    else:
                        # Candidate speech faded before the debounce
                        # elapsed — a cough, a click, a "hmm". Ignore.
                        self._candidate_start_ts = None

                # Accumulate audio while speaking (or while a candidate is
                # pending, so the pre-debounce audio isn't lost).
                if self.is_speaking or self._candidate_start_ts is not None:
                    self._utterance_chunks.append(audio_int16)
        except Exception as e:
            print(f"[VAD] audio loop stopped: {e}")

    async def listen_loop(self):
        """
        Async entry point for the orchestrator. Runs the blocking audio loop
        in a worker thread and resolves to None when the loop stops.
        """
        self._loop = asyncio.get_running_loop()
        self._load_model()
        print("VAD Listening... (Press Ctrl+C to stop)")
        # Run the blocking capture loop off the event loop thread.
        await self._loop.run_in_executor(None, self._audio_thread)

    def listen(self):
        """Blocking, synchronous entry point (for standalone use / testing)."""
        self._load_model()
        print("VAD Listening... (Press Ctrl+C to stop)")
        try:
            self._audio_thread()
        except KeyboardInterrupt:
            self.stop()

    # ------------------------------------------------------------------ #
    # Utterance access
    # ------------------------------------------------------------------ #
    def get_last_utterance(self) -> bytes:
        """
        Return the raw int16 PCM bytes of the most recent utterance.

        Returns an empty bytes object if no speech has been captured yet.
        """
        if not self._utterance_chunks:
            return b""
        return np.concatenate(self._utterance_chunks).tobytes()

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()


# Example usage for testing
if __name__ == "__main__":
    def handle_start():
        print(">>> Triggering ASR / Interrupting TTS...")

    def handle_end():
        print(">>> Finalizing Transcription...")

    vad = VADManager(on_speech_start=handle_start, on_speech_end=handle_end)
    vad.listen()
