"""Voice activity detection for the RPi service.

Prefers webrtcvad (tiny, real-time on RPi). Falls back to a simple
energy+zero-crossing detector when webrtcvad is not installed, so the
service runs anywhere for development.
"""
import numpy as np

try:
    import webrtcvad
    HAVE_WEBRTC = True
except ImportError:
    HAVE_WEBRTC = False


class VAD:
    """Frame-based VAD. feed(frame_bytes) -> bool (speech?).

    Frames must be 16-bit PCM mono at 8/16/32/48 kHz, 10/20/30 ms long.
    """

    def __init__(self, sr=16000, frame_ms=30, aggressiveness=2):
        if sr not in (8000, 16000, 32000, 48000) or frame_ms not in (10, 20, 30):
            raise ValueError("webrtcvad needs sr in {8k,16k,32k,48k}, "
                             "frame in {10,20,30} ms")
        self.sr = sr
        self.frame_ms = frame_ms
        self.frame_bytes = sr * frame_ms // 1000 * 2
        if HAVE_WEBRTC:
            self.vad = webrtcvad.Vad(aggressiveness)
        else:
            self.vad = None
            self._thr = 0.01  # energy threshold (tune per mic)

    def frame_bytes_len(self):
        return self.frame_bytes

    def feed(self, frame: bytes) -> bool:
        if self.vad is not None:
            return self.vad.is_speech(frame, self.sr)
        x = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
        return float(np.abs(x).mean()) / 32768.0 > self._thr


def collect_utterance(vad, frame_iter, sr, max_s=6.0, silence_ms=700):
    """Read frames from frame_iter() until end-of-speech.

    Returns (audio_float32, sr) for one utterance, or (None, sr) on timeout.
    frame_iter yields bytes of exactly vad.frame_bytes_len().
    """
    import time
    frame_s = vad.frame_ms / 1000.0
    max_frames = int(max_s / frame_s)
    silence_frames = int(silence_ms / vad.frame_ms)
    buf, voiced, started, silent_run = [], 0, False, 0
    t0 = time.time()
    for i, frame in enumerate(frame_iter()):
        if i >= max_frames or (time.time() - t0) > max_s:
            break
        speech = vad.feed(frame)
        if speech:
            started = True
            silent_run = 0
            voiced += 1
        elif started:
            silent_run += 1
        if started:
            buf.append(frame)
        if started and silent_run >= silence_frames:
            break
    if not buf or voiced < 3:  # need ~90 ms of speech minimum
        return None, sr
    pcm = b"".join(buf)
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return x, sr
