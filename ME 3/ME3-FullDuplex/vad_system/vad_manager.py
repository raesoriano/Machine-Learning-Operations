"""
VAD Manager for ME3-FullDuplex.

Based on the ME 3 VADManager (Silero VAD), with two additions:

1. Acoustic echo cancellation (AEC) — the TTS audio that is sent to the
   speaker is fed into an AEC filter (see `aec.py`), which subtracts it from
   the live mic stream *before* the VAD runs. The VAD therefore sees only the
   user's voice and can run continuously, even while the agent is talking.
   This is what makes barge-in real: the mic is never muted, so the agent
   genuinely listens while it speaks.

   The AEC backend is SpeexDSP when the system library is present, and a
   pure-NumPy NLMS filter otherwise — so there is no hard dependency.

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

# AEC lives in the ME3-FullDuplex package root (one level up from vad_system/).
import os as _os
import sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_PARENT = _os.path.dirname(_HERE)
if _PARENT not in _sys.path:
    _sys.path.insert(0, _PARENT)
from aec import AEC  # noqa: E402

# Canonical Silero VAD source (was incorrectly 'snyprow/silero-vad' before).
SILERO_REPO = "snakers4/silero-vad"
SILERO_MODEL = "silero_vad"
# Cache the model so repeated instantiations don't re-download it.
SILERO_CACHE = "silero_vad"


class VADManager:
    """
    VAD Manager using Silero VAD for high-accuracy voice detection on
    Raspberry Pi, with acoustic echo cancellation and barge-in debounce.

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

        # Acoustic echo cancellation. The TTS playback thread calls
        # `feed_reference()` with the audio it writes to the speaker; the
        # audio thread calls `process()` to strip that echo out of the mic
        # before running the VAD. The VAD is therefore never deafened while
        # the agent talks — it listens continuously (real full-duplex).
        self.aec = AEC(sample_rate=sample_rate, block_size=chunk_size)

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
    # Acoustic echo cancellation
    # ------------------------------------------------------------------ #
    def feed_reference(self, tts_int16: np.ndarray):
        """
        Feed the TTS audio just written to the speaker into the AEC.

        Called from the TTS playback thread. The AEC resamples this to the
        mic rate and delays it so it lines up with the mic capture.
        """
        self.aec.reference(tts_int16)

    @property
    def aec_backend(self) -> str:
        """Which AEC backend is active: 'speexdsp' or 'nlms'."""
        return self.aec.backend

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
        """Blocking audio capture + AEC + VAD inference, worker thread."""
        try:
            while True:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_int16 = np.frombuffer(data, dtype=np.int16)

                # Acoustic echo cancellation: subtract the agent's own voice
                # (the TTS reference) from the mic. The VAD runs on the clean
                # signal, so it stays live even while the agent is speaking —
                # this is what makes barge-in possible.
                clean_int16 = self.aec.process(audio_int16)

                speech_prob = self._process_chunk(clean_int16)

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

                # Accumulate the *clean* (echo-cancelled) audio while speaking
                # (or while a candidate is pending, so the pre-debounce audio
                # isn't lost). The ASR gets the user's voice without the
                # agent's own TTS bleeding in.
                if self.is_speaking or self._candidate_start_ts is not None:
                    self._utterance_chunks.append(clean_int16)
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
