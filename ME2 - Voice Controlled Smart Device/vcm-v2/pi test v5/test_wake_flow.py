"""Synthetic test of the wake-word gate flow (no mic needed).

Feeds the real WakeGate a 48 kHz stream built from:
  1. 1.0 s of noise
  2. "hey rhasspy" (espeak, resampled to 48k)  -> should fire the wake word
  3. 0.5 s of the wake word's own tail (more speech, >= 300 ms)
  4. 0.3 s silence
  5. "pause the music" (espeak, resampled to 48k)  -> should be the command
  6. 1.0 s of noise

Asserts:
  - the wake word fires
  - the wake word's tail is NOT returned as a command (the bug)
  - the real command IS captured
  - noise alone never fires the wake word
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcm_pi_v5 as v5          # noqa: E402


def synth(text, sr_out=48000, dur_pad=0.0):
    """espeak a phrase, resample to sr_out, return float32 mono."""
    from scipy.signal import resample_poly
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    subprocess.run(
        ["espeak", "-v", "en-us", "-s", "150", "-w", path, text],
        check=True, capture_output=True)
    import wave
    with wave.open(path) as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    os.unlink(path)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr_out != sr:
        a = resample_poly(a, sr_out, sr)
    if dur_pad:
        a = np.concatenate([a, np.zeros(int(sr_out * dur_pad), np.float32)])
    return a


def noise(sr, dur, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(sr * dur)) * 0.02).astype(np.float32)


def feed(gate, stream, sr):
    """Feed a 48k stream through the gate in 30 ms frames; return events."""
    frame = int(sr * 0.030)
    events = []
    t0 = time.time()
    for i in range(0, len(stream) - frame, frame):
        f = stream[i:i + frame]
        ev = gate.push(f, t0 + i / sr)
        if ev:
            events.append((i / sr, ev))
    return events


def main():
    sr = 48000
    wake = v5.WakeWord(v5.WAKEWORD_MODEL, threshold=0.5)
    vad = v5.VAD(sr, gate=0.05)
    gate = v5.WakeGate(sr, wake, vad)

    # --- build the stream -------------------------------------------------
    wake_word = synth("hey rhasspy", sr)
    tail = synth("rhasspy", sr)            # the tail of the wake word
    cmd = synth("pause the music", sr)
    stream = np.concatenate([
        noise(sr, 1.0),
        wake_word,
        tail,                               # >= 300 ms of speech right after
        np.zeros(int(sr * 0.3), np.float32),
        cmd,
        noise(sr, 1.0),
    ])

    events = feed(gate, stream, sr)
    print("events:", [(round(t, 2), e) for t, e in events])

    wake_ts = [t for t, e in events if e == "wake"]
    cmd_ts = [t for t, e in events if e == "command"]

    assert wake_ts, "wake word did NOT fire"
    print(f"  wake word fired at t={wake_ts[0]:.2f}s  OK")

    assert cmd_ts, "real command was NOT captured"
    print(f"  command captured at t={cmd_ts[0]:.2f}s  OK")

    # the command must come AFTER the wake word + the tail, not be the tail
    gap = cmd_ts[0] - wake_ts[0]
    assert gap > 0.6, f"command at {gap:.2f}s after wake -- that is the tail, not a real command"
    print(f"  command is {gap:.2f}s after wake (tail excluded)  OK")

    # the captured audio should be the command, not the tail
    a = gate.last_audio
    print(f"  captured audio: {len(a)/16000:.2f}s at 16kHz")
    assert len(a) >= sr * 0.3, "captured audio too short"

    # --- noise alone must never fire --------------------------------------
    gate2 = v5.WakeGate(sr, v5.WakeWord(v5.WAKEWORD_MODEL, threshold=0.5),
                        v5.VAD(sr, gate=0.05))
    ev2 = feed(gate2, noise(sr, 5.0, seed=1), sr)
    assert not ev2, f"noise fired the wake word: {ev2}"
    print("  5s of noise -> no events  OK")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
