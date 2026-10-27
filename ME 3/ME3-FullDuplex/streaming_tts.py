"""
Streaming TTS for ME3-FullDuplex.

Piper is invoked with `--output-raw` so raw 16-bit PCM is written to
stdout as it is produced. A playback thread reads those bytes and writes
them to the PyAudio output stream in 30 ms frames — the first sentence
starts playing while the LLM is still generating the rest.

The playback thread also feeds *every* frame it writes to the speaker into
the VAD's acoustic echo canceller (`vad.feed_reference`). That lets the VAD
subtract the agent's own voice from the mic and keep listening while TTS is
playing — which is what makes barge-in real. (The old "echo gate", which
muted the VAD during playback, is gone.)

Fallback: if the Piper binary doesn't support `--output-raw` (Piper < 1.2),
each sentence is synthesized to a temp WAV and played whole. Streaming
still works at sentence granularity — only intra-sentence streaming is lost.

Piper paths come from the same env vars as ME 3 (tts_server/synthesize.py):
PIPER_BINARY, PIPER_MODEL, PIPER_CONFIG, PIPER_LD_LIBRARY_PATH.
"""

import os
import queue
import subprocess
import tempfile
import threading
import time
import wave
from typing import Callable, Optional

import numpy as np
import pyaudio

# --- Piper configuration (same env vars as ME 3) ------------------------ #
PIPER_BINARY = os.environ.get("PIPER_BINARY", "/home/raesoriano/piper/piper")
PIPER_MODEL = os.environ.get("PIPER_MODEL", "/home/raesoriano/piper/en_US-lessac-medium.onnx")
PIPER_CONFIG = os.environ.get(
    "PIPER_CONFIG", "/home/raesoriano/piper/en_US-lessac-medium.onnx.json"
)
PIPER_LD_LIBRARY_PATH = os.environ.get("PIPER_LD_LIBRARY_PATH", "/home/raesoriano/piper")

# --- Playback tuning ---------------------------------------------------- #
SAMPLE_RATE = 22050          # Piper's native output rate
PLAYBACK_FRAME_MS = 30       # PyAudio write chunk (ms)

# Piper's output sample rate is model-dependent; the lessac models are
# 22050 Hz. If the model differs, the config json carries "sample_rate".
def _piper_sample_rate() -> int:
    try:
        import json
        with open(PIPER_CONFIG) as f:
            cfg = json.load(f)
        return int(cfg.get("sample_rate", SAMPLE_RATE))
    except Exception:
        return SAMPLE_RATE


def _piper_env() -> dict:
    env = dict(os.environ)
    if PIPER_LD_LIBRARY_PATH:
        env["LD_LIBRARY_PATH"] = PIPER_LD_LIBRARY_PATH
    return env


def _supports_output_raw() -> bool:
    """Probe the Piper binary once for --output-raw support."""
    global _RAW_SUPPORTED
    if "_RAW_SUPPORTED" not in globals():
        try:
            r = subprocess.run(
                [PIPER_BINARY, "--help"],
                capture_output=True, text=True, env=_piper_env(), timeout=10,
            )
            _RAW_SUPPORTED = "--output-raw" in (r.stdout + r.stderr)
        except Exception:
            _RAW_SUPPORTED = False
    return _RAW_SUPPORTED


class StreamingTTS:
    """
    Synthesizes sentences with Piper and plays them as they arrive.

    Usage:
        tts = StreamingTTS(vad)            # vad: VADManager (for AEC reference)
        tts.start()
        tts.enqueue("First sentence.")     # from the LLM sentence generator
        tts.enqueue("Second sentence.")
        tts.finish()                       # no more sentences coming
        tts.wait()                         # block until playback drains

    Barge-in:
        stop() kills the Piper subprocess, flushes the queue, and stops
        playback. The orchestrator calls this on a confirmed barge-in.
    """

    def __init__(self, vad=None, sample_rate: Optional[int] = None):
        self.vad = vad
        self.sample_rate = sample_rate or _piper_sample_rate()
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._playing = threading.Event()
        self._raw = _supports_output_raw()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream = None
        self._frames_played = 0
        # Telemetry hook: called with (sentence_index, first_frame_ts).
        self.on_first_frame: Optional[Callable] = None
        self._sentence_idx = 0
        self._first_frame_done = False
        # AEC reference feed: the TTS rate, so the VAD's AEC can resample the
        # reference to the mic rate. (The AEC itself lives on the VAD.)
        self._ref_rate = self.sample_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def start(self):
        """Open the output stream and start the playback thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._first_frame_done = False
        self._sentence_idx = 0
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=int(self.sample_rate * PLAYBACK_FRAME_MS / 1000),
        )
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def enqueue(self, sentence: str):
        """Queue a sentence for synthesis + playback (non-blocking)."""
        if not sentence or not sentence.strip():
            return
        self._q.put(sentence.strip())

    def finish(self):
        """Signal that no more sentences are coming."""
        self._q.put(None)

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Block until playback drains. Returns True if fully drained,
        False on timeout.
        """
        if self._thread is None:
            return True
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def is_playing(self) -> bool:
        return self._playing.is_set()

    def stop(self):
        """
        Kill everything: stop Piper, drop queued sentences, stop playback.
        Used for barge-in.

        No echo-gate cleanup is needed anymore — the VAD's AEC keeps running
        continuously, so there is nothing to "re-open" after playback stops.
        """
        self._stop_event.set()
        # Drain the queue so the playback thread doesn't pick up leftovers.
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(None)
        self._playing.clear()

    def close(self):
        """Release the audio device (call once at shutdown)."""
        self.stop()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
        self._stream = None
        self._pa = None

    # ------------------------------------------------------------------ #
    # Playback loop (worker thread)
    # ------------------------------------------------------------------ #
    def _playback_loop(self):
        frame_bytes = int(self.sample_rate * PLAYBACK_FRAME_MS / 1000) * 2
        piper_proc: Optional[subprocess.Popen] = None
        pcm_buf = b""

        def _close_piper():
            nonlocal piper_proc
            if piper_proc is not None:
                try:
                    piper_proc.stdin.close()
                except Exception:
                    pass
                try:
                    piper_proc.wait(timeout=5)
                except Exception:
                    piper_proc.kill()
                piper_proc = None

        try:
            while not self._stop_event.is_set():
                item = self._q.get()
                if item is None:
                    # No more sentences: close Piper, drain remaining PCM.
                    _close_piper()
                    break
                if self._stop_event.is_set():
                    break

                # Synthesize this sentence.
                if self._raw:
                    piper_proc = self._spawn_piper_raw(item)
                    if piper_proc is None:
                        continue
                else:
                    pcm_buf += self._synthesize_wav(item)
                    _close_piper()

                # Play what we have. The VAD's AEC is fed the exact frames we
                # write (in _write_frame), so it can cancel our own voice and
                # keep listening while we speak. No gate to open/close.
                self._playing.set()

                if self._raw:
                    self._play_raw_stream(piper_proc, frame_bytes)
                    _close_piper()
                else:
                    self._play_pcm(pcm_buf, frame_bytes)
                    pcm_buf = b""

        except Exception as e:
            print(f"[TTS] playback loop error: {e}")
        finally:
            _close_piper()
            self._playing.clear()

    def _spawn_piper_raw(self, sentence: str) -> Optional[subprocess.Popen]:
        """Start Piper with --output-raw; returns the Popen or None."""
        try:
            proc = subprocess.Popen(
                [PIPER_BINARY, "--model", PIPER_MODEL,
                 "--config", PIPER_CONFIG, "--output-raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_piper_env(),
            )
            proc.stdin.write(sentence.encode("utf-8"))
            return proc
        except Exception as e:
            print(f"[TTS] piper spawn failed: {e}")
            return None

    def _play_raw_stream(self, proc: subprocess.Popen, frame_bytes: int):
        """Read raw PCM from Piper's stdout and play it frame by frame."""
        while not self._stop_event.is_set():
            chunk = proc.stdout.read(frame_bytes)
            if not chunk:
                break
            self._write_frame(chunk)
        # Drain any partial final frame.
        tail = proc.stdout.read()
        if tail:
            self._write_frame(tail)

    def _synthesize_wav(self, sentence: str) -> bytes:
        """Fallback: synthesize a sentence to a temp WAV, return its PCM."""
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            proc = subprocess.run(
                [PIPER_BINARY, "--model", PIPER_MODEL,
                 "--config", PIPER_CONFIG, "--output_file", path],
                input=sentence, capture_output=True, text=True,
                env=_piper_env(), timeout=30,
            )
            if proc.returncode != 0:
                print(f"[TTS] piper failed: {proc.stderr[:200]}")
                return b""
            with wave.open(path, "rb") as w:
                if w.getframerate() != self.sample_rate:
                    # Simple linear resample (good enough for fallback).
                    raw = w.readframes(w.getnframes())
                    return self._resample(raw, w.getframerate(), self.sample_rate)
                return w.readframes(w.getnframes())
        except Exception as e:
            print(f"[TTS] wav synth failed: {e}")
            return b""
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def _resample(self, pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
        """Linear resample of 16-bit mono PCM (fallback path only)."""
        if src_rate == dst_rate:
            return pcm
        import array
        samples = array.array("h")
        samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
        n_out = int(len(samples) * dst_rate / src_rate)
        out = array.array("h", bytes(2 * n_out))
        for i in range(n_out):
            pos = i * src_rate / dst_rate
            i0 = int(pos)
            i1 = min(i0 + 1, len(samples) - 1)
            frac = pos - i0
            out[i] = int(samples[i0] * (1 - frac) + samples[i1] * frac)
        return out.tobytes()

    def _play_pcm(self, pcm: bytes, frame_bytes: int):
        """Play a complete PCM buffer in frames."""
        for i in range(0, len(pcm), frame_bytes):
            if self._stop_event.is_set():
                break
            self._write_frame(pcm[i:i + frame_bytes])

    def _write_frame(self, chunk: bytes):
        """Write one frame to the output stream, padding to frame size."""
        if self._stream is None:
            return
        if len(chunk) % 2:
            chunk += b"\x00"
        try:
            self._stream.write(chunk, exception_on_overflow=False)
            self._frames_played += len(chunk)
            # Feed the exact audio we just sent to the speaker into the VAD's
            # AEC, so it can subtract our own voice from the mic. This is the
            # reference signal for echo cancellation.
            if self.vad is not None:
                try:
                    self.vad.feed_reference(np.frombuffer(chunk, dtype=np.int16))
                except Exception:
                    pass
            if not self._first_frame_done:
                self._first_frame_done = True
                if self.on_first_frame is not None:
                    try:
                        self.on_first_frame(self._sentence_idx, time.monotonic())
                    except Exception:
                        pass
        except Exception as e:
            print(f"[TTS] write error: {e}")
