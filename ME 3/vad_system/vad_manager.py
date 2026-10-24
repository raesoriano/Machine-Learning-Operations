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
    VAD Manager using Silero VAD for high-accuracy voice detection on Raspberry Pi.

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
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

        # Model is loaded lazily (see _load_model) so importing/instantiating
        # this class never triggers a network download.
        self.model = None
        self.utils = None

        self.is_speaking = False
        self._utterance_chunks = []

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
                speech_prob = self._process_chunk(audio_int16)

                if speech_prob > 0.5 and not self.is_speaking:
                    self.is_speaking = True
                    self._utterance_chunks = []  # reset buffer for new utterance
                    print("[VAD] Speech Started")
                    self._fire(self.on_speech_start)

                elif speech_prob < 0.3 and self.is_speaking:
                    self.is_speaking = False
                    print("[VAD] Speech Ended")
                    self._fire(self.on_speech_end)

                # Accumulate audio while speaking so we can hand it to ASR.
                if self.is_speaking:
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
