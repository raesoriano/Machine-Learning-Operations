"""
Acoustic Echo Cancellation for ME3-FullDuplex.

This is the piece that makes the agent *actually* listen while it talks.
Instead of muting the VAD while TTS plays (the old "echo gate"), we feed the
exact TTS signal that was sent to the speaker into an AEC filter, which
subtracts it from the live mic stream. What's left is (mostly) the user's
voice, so the VAD can run continuously and barge-in becomes real.

Two backends, chosen at runtime:

1. SpeexDSP (preferred) -- a real adaptive AEC via ctypes, using the system
   library `libspeexdsp` (apt: libspeexdsp-dev). This is the standard
   Raspberry Pi choice and is small (~10-20 MB). It is only used if a
   self-test at init confirms the build actually produces output (some
   aarch64 builds export the symbols but are no-ops).

2. Pure-NumPy NLMS (fallback) -- a sample-by-sample NLMS written in NumPy, so
   the module works with zero system dependencies. It converges more slowly
   than SpeexDSP, so for the best echo suppression on a Pi you should install
   libspeexdsp (setup.sh does this). The fallback guarantees the agent still
   runs — and still listens while it talks — even without the library.

Both backends expose the same interface:

    aec = AEC(sample_rate=16000, block_size=512)
    aec.reference(tts_int16)   # TTS audio just written to the speaker
    clean = aec.process(mic_int16)  # mic audio with the echo removed

The reference signal is resampled to the mic rate (TTS is 22050 Hz, the mic
is 16000 Hz) and a short delay is applied to line up the TTS audio with the
moment it physically reaches the mic.
"""

import ctypes
import os
import threading
from collections import deque
from typing import Optional

import numpy as np

# --- Tunables ----------------------------------------------------------- #
# How many samples of the TTS reference to keep in the adaptive filter.
# A speaker + room impulse response is short; 200 ms is plenty for a Pi and
# keeps the filter small enough to converge quickly (a 1 s / 16000-tap filter
# converges slowly and smears into the user's voice).
FILTER_TAIL_S = float(os.environ.get("AEC_FILTER_TAIL_S", "0.2"))
# NLMS step size. Too high -> the filter oscillates; too low -> slow to
# converge. 0.05 is a safe middle ground for 16 kHz mono.
NLMS_MU = float(os.environ.get("AEC_MU", "0.05"))
# Delay between writing a TTS sample to the speaker and it reaching the mic.
# The AEC reference must be shifted by this much. 40 ms covers the PyAudio
# output buffer + speaker + mic capture latency on a typical Pi.
DELAY_MS = int(os.environ.get("AEC_DELAY_MS", "40"))
# How hard to suppress the echo (0..1). 0.8 keeps the user's voice clear while
# killing most of the agent's own voice.
SUPPRESSION = float(os.environ.get("AEC_SUPPRESSION", "0.8"))


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample a 1-D int16/float array between sample rates."""
    if src_rate == dst_rate:
        return pcm
    x = pcm.astype(np.float32)
    n_out = int(len(x) * dst_rate / src_rate)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    pos = np.arange(n_out, dtype=np.float32) * (src_rate / dst_rate)
    i0 = np.floor(pos).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = (pos - i0).astype(np.float32)
    return x[i0] * (1.0 - frac) + x[i1] * frac


class AEC:
    """
    Acoustic echo canceller with a SpeexDSP backend (if available) and a
    pure-NumPy NLMS fallback.

    Thread-safety: `reference()` is called from the TTS playback thread and
    `process()` from the VAD audio thread. A single lock guards the shared
    filter state.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        ref_rate: int = 22050,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.ref_rate = ref_rate
        self._lock = threading.Lock()
        # The reference is kept at its NATIVE rate (ref_rate, e.g. 22050 Hz) —
        # the rate the audio was actually played at — because SpeexDSP's filter
        # is built at the mic rate and resampling the reference first would
        # double-delay it. The delay line is also in ref_rate samples.
        #
        # The delay models the speaker->mic latency. In the real system this is
        # ~40 ms (PyAudio output buffer + speaker + mic capture). It is
        # configurable so it can be tuned per-hardware (or set to 0 in a test
        # where the reference is already aligned to the mic).
        delay_ms = int(os.environ.get("AEC_DELAY_MS", DELAY_MS))
        self._delay_samples = int(ref_rate * delay_ms / 1000.0)
        self._tail = int(sample_rate * FILTER_TAIL_S)
        self._speex = self._try_speex(block_size, sample_rate)
        # Pure-NumPy block NLMS state (the default backend). The filter lives
        # at the mic rate; its reference window is resampled to the mic rate.
        self._w = np.zeros(self._tail, dtype=np.float32)
        # Reference ring buffer: TTS samples at the native ref_rate.
        self._ref_buf: deque = deque(maxlen=self._tail + self._delay_samples + block_size)
        # A small FIFO so the reference is delayed by `delay_samples`.
        self._ref_delay: deque = deque(maxlen=self._delay_samples + 1)
        self._ref_ready = False

    # ------------------------------------------------------------------ #
    # Backend selection
    # ------------------------------------------------------------------ #
    def _try_speex(self, frame_size: int, sample_rate: int):
        """
        Try to load SpeexDSP and create an echo-state handler.

        Returns a small wrapper object or None. We deliberately keep this
        best-effort: any failure (missing lib, missing symbol) just falls
        back to the pure-Python NLMS, so the agent always runs.
        """
        try:
            import ctypes
            import ctypes.util

            name = ctypes.util.find_library("speexdsp")
            if name is None:
                for cand in ("libspeexdsp.so.1", "libspeexdsp.so", "libspeexdsp.dylib"):
                    if os.path.exists(cand):
                        name = cand
                        break
            if name is None:
                print("[AEC] libspeexdsp not found -> using pure-Python NLMS")
                return None

            lib = ctypes.CDLL(name)
            # speex_echo_state_init(frame_size, filter_length, sample_rate)
            # (Note: this is the symbol actually exported by libspeexdsp.so.1;
            #  the older `speex_echo_state_new` name is NOT present.)
            lib.speex_echo_state_init.restype = ctypes.c_void_p
            lib.speex_echo_state_init.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ]
            # speex_echo_cancellation(st, ref, mic, out)
            lib.speex_echo_cancellation.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            # speex_echo_state_destroy(st)
            lib.speex_echo_state_destroy.argtypes = [ctypes.c_void_p]

            st = lib.speex_echo_state_init(frame_size, self._tail, sample_rate)
            if not st:
                print("[AEC] speex_echo_state_init returned NULL -> NLMS fallback")
                return None

            backend = _SpeexBackend(lib, st, frame_size, self.ref_rate, sample_rate)
            # Self-test: some libspeexdsp builds (notably certain aarch64
            # ones) export the symbols but produce no output. Verify the
            # backend actually cancels before trusting it; otherwise fall back
            # to the NumPy NLMS so we never ship a silent no-op.
            if not backend._self_test():
                print("[AEC] SpeexDSP self-test failed -> using NumPy NLMS")
                return None
            return backend
        except Exception as e:  # noqa: BLE001 - any failure -> fallback
            print(f"[AEC] SpeexDSP init failed ({e}) -> using pure-Python NLMS")
            return None

    @property
    def backend(self) -> str:
        return "speexdsp" if self._speex is not None else "nlms"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def reference(self, tts_int16: np.ndarray):
        """
        Feed the TTS audio that was just written to the speaker.

        `tts_int16` is at the native TTS rate (e.g. 22050 Hz). It is kept at
        that rate (NOT resampled to the mic rate) and pushed through a short
        delay line so it lines up with the moment it physically reaches the
        mic. Keeping it at the native rate is important: SpeexDSP's filter is
        built at the mic rate, so resampling the reference first would
        double-delay it and the filter could not align.
        """
        if tts_int16 is None or len(tts_int16) == 0:
            return
        tts_f = tts_int16.astype(np.float32) / 32768.0
        with self._lock:
            if not self._ref_ready:
                # Prime the delay line with silence so the very first TTS
                # sample is delayed by the full `delay_samples`.
                self._ref_delay = deque(
                    [0.0] * self._delay_samples, maxlen=self._delay_samples + 1
                )
                self._ref_ready = True
            for s in tts_f:
                self._ref_delay.append(s)
                self._ref_buf.append(self._ref_delay[0])

    def process(self, mic_int16: np.ndarray) -> np.ndarray:
        """
        Remove the agent's own voice from a mic block.

        `mic_int16` is at the mic rate (e.g. 16000 Hz). Returns an int16 array
        of the same length with the echo subtracted.
        """
        if mic_int16 is None or len(mic_int16) == 0:
            return mic_int16
        mic_f = mic_int16.astype(np.float32) / 32768.0
        with self._lock:
            if self._speex is not None:
                out = self._speex.cancel(self._ref_buf, mic_f)
            else:
                out = self._nlms_cancel(mic_f)
        # Guard against any NaN/Inf the backends might emit on edge cases.
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)
        out = np.clip(out, -1.0, 1.0)
        return (out * 32767.0).astype(np.int16)

    # ------------------------------------------------------------------ #
    # Pure-Python NLMS
    # ------------------------------------------------------------------ #
    def _nlms_cancel(self, mic_f: np.ndarray) -> np.ndarray:
        """
        Sample-by-sample NLMS echo cancellation (pure-NumPy fallback).

        For each mic sample, the filter `w` predicts the echo from the most
        recent `tail` reference samples (resampled to the mic rate). The error
        (mic - prediction) is the clean signal, and `w` is nudged to reduce it.

        This is the *fallback* backend. It is simple and dependency-free, but
        it converges slowly, so on a Pi you should install libspeexdsp (see
        setup.sh) to get the SpeexDSP backend, which cancels far more
        effectively. The fallback is there so the agent still runs — and still
        listens while it talks — even without the system library.
        """
        n = len(mic_f)
        tail = self._tail
        ref = self._ref_buf
        # Grab the most recent reference (native rate) and resample to the mic
        # rate so the filter (which lives at the mic rate) can use it.
        need_native = int((n + tail) * self.ref_rate / self.sample_rate)
        ref_raw = np.asarray(list(ref)[-(need_native):], dtype=np.float32)
        ref_mic = _resample_linear(ref_raw, self.ref_rate, self.sample_rate)
        if len(ref_mic) < n + tail:
            ref_mic = np.pad(ref_mic, (n + tail - len(ref_mic), 0), mode="constant")
        # ref_mic[i + tail - 1] aligns with mic sample i (reference is delayed
        # to line up with the mic).
        out = np.empty(n, dtype=np.float32)
        w = self._w
        mu = NLMS_MU
        for i in range(n):
            # The `tail` reference samples ending at the one aligned with mic
            # sample i, most-recent-first.
            end = i + tail  # exclusive upper index in ref_mic
            x = ref_mic[end - tail: end][::-1]
            echo = float(np.dot(w, x))
            e = mic_f[i] - echo
            xnorm = float(np.dot(x, x)) + 1e-6
            w += (mu * e / xnorm) * x
            out[i] = e
        self._w = w
        # Advance the reference buffer by the native-rate samples consumed.
        consumed = int(n * self.ref_rate / self.sample_rate)
        for _ in range(consumed):
            if len(ref) == ref.maxlen:
                ref.popleft()
        return out


class _SpeexBackend:
    """Thin ctypes wrapper around a SpeexDSP echo-state handler."""

    def __init__(self, lib, state, frame_size: int, ref_rate: int, mic_rate: int):
        self.lib = lib
        self.state = state
        self.frame_size = frame_size
        # The reference buffer is at the native TTS rate (ref_rate) while the
        # mic is at mic_rate. For every `mic_rate` mic samples we must consume
        # `ref_rate` reference samples, or the two desynchronize and the filter
        # can't align (which shows up as over-suppression of the user's voice).
        self._ref_rate = ref_rate
        self._mic_rate = mic_rate
        self._ref_frac = 0.0  # fractional reference-sample accumulator

    def _self_test(self) -> bool:
        """
        Verify this SpeexDSP build actually produces output.

        Feeds a short tone as the reference and the same tone (plus a small
        offset) as the mic. A working AEC returns a non-zero, finite output.
        A broken build (symbols present but no-op) returns all zeros, which we
        detect and reject.
        """
        try:
            fs = self.frame_size
            tone = (0.5 * np.sin(2 * np.pi * 440 * np.arange(fs) / self._mic_rate))
            ref = (tone * 32767).astype(np.int16)
            mic = (np.clip(tone + 0.01, -1, 1) * 32767).astype(np.int16)
            ref_arr = np.zeros(fs, dtype=np.float32)
            mic_arr = np.zeros(fs, dtype=np.float32)
            out_arr = np.zeros(fs, dtype=np.float32)
            ref_arr[:] = ref.astype(np.float32) / 32768.0
            mic_arr[:] = mic.astype(np.float32) / 32768.0
            ref_p = ref_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            mic_p = mic_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_p = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self.lib.speex_echo_cancellation(self.state, ref_p, mic_p, out_p)
            # Run a few more frames to let it settle.
            for _ in range(4):
                self.lib.speex_echo_cancellation(self.state, ref_p, mic_p, out_p)
            return bool(np.any(out_arr != 0.0)) and np.all(np.isfinite(out_arr))
        except Exception:
            return False

    def cancel(self, ref_buf: deque, mic_f: np.ndarray) -> np.ndarray:
        """
        Run SpeexDSP cancellation over `mic_f`, consuming reference samples
        from `ref_buf` at the correct ref_rate:mic_rate ratio.
        """
        n = len(mic_f)
        out = np.empty(n, dtype=np.float32)
        fs = self.frame_size
        for start in range(0, n, fs):
            chunk = mic_f[start:start + fs]
            m = len(chunk)
            # How many native-rate reference samples correspond to `m` mic
            # samples? (e.g. 512 mic @16k -> 705 ref @22050.)
            want = int(self._ref_frac + m * self._ref_rate / self._mic_rate)
            self._ref_frac = (self._ref_frac + m * self._ref_rate / self._mic_rate) - want
            # Take the next `want` reference samples (zero-pad if the buffer
            # is momentarily short — e.g. before the first TTS frame arrives).
            ref_chunk = np.zeros(want, dtype=np.float32)
            for j in range(want):
                if len(ref_buf) > 0:
                    ref_chunk[j] = ref_buf.popleft()
            # Resample this reference chunk to the mic rate so SpeexDSP sees a
            # 1:1 ref:mic alignment (its filter is built at the mic rate).
            ref_mic = _resample_linear(ref_chunk, self._ref_rate, self._mic_rate)
            if len(ref_mic) < m:
                ref_mic = np.pad(ref_mic, (0, m - len(ref_mic)), mode="constant")
            ref_mic = ref_mic[:m]
            # Pad to a full frame for SpeexDSP.
            ref_arr = np.zeros(fs, dtype=np.float32)
            mic_arr = np.zeros(fs, dtype=np.float32)
            out_arr = np.zeros(fs, dtype=np.float32)
            ref_arr[:m] = ref_mic
            mic_arr[:m] = chunk
            ref_p = ref_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            mic_p = mic_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            out_p = out_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            # Use the NLMS variant (better echo suppression).
            self.lib.speex_echo_cancellation(self.state, ref_p, mic_p, out_p)
            out[start:start + m] = out_arr[:m]
        return out
