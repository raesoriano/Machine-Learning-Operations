#!/usr/bin/env python3
"""Generate the wake-word self-test fixture (espeak 'hey rhasspy' -> 16 kHz
int16 mono wav) and verify it fires the model through the production path."""
import os
import subprocess
import sys
import wave

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vcm_pi_v5 import WakeWord  # noqa: E402

raw = "/tmp/hello_raw.wav"
subprocess.run(["espeak", "-v", "en-us", "-s", "150", "-w", raw,
                "hey rhasspy"], check=True, capture_output=True)
with wave.open(raw) as w:
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
a16 = resample_poly(a, 16000, sr)
print("raw sr:", sr, "dur:", round(len(a) / sr, 2), "s -> 16k:", round(len(a16) / 16000, 2), "s")

ww = WakeWord(os.path.join(HERE, "wakeword", "hey_rhasspy_v0.1.onnx"), threshold=0.5)
pcm = (np.clip(a16, -1, 1) * 32767).astype(np.int16)
peak = 0.0
for i in range(0, len(pcm) - 1279, 1280):
    peak = max(peak, ww.push(pcm[i:i + 1280]))
print("peak score:", round(peak, 4), "fires:", peak >= 0.5)
assert peak >= 0.5, "self-test fixture does not fire the model"

import soundfile as sf
dst = os.path.join(HERE, "wakeword", "selftest_hey_rhasspy.wav")
sf.write(dst, (np.clip(a16, -1, 1) * 32767).astype(np.int16), 16000)
print("saved", dst, os.path.getsize(dst), "bytes")
