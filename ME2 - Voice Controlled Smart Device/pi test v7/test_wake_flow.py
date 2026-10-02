"""Synthetic test of the wake-word gate flow (no mic needed).

Feeds the real WakeGate a 48 kHz stream built from:
  1. 1.0 s of noise
  2. "hey rhasspy" (espeak, resampled to 48k)  -> should fire the wake word
  3. the wake word's own tail (more speech, >= 300 ms)
  4. 0.3 s silence
  5. "pause the music" (espeak, resampled to 48k)  -> should be the command
  6. 1.0 s of noise

The test simulates the main thread: when the gate reports 'wake', it plays
the cue (0.67 s of muted time) and then calls gate.cue_done() to arm the
command window -- exactly what run_mic does.

Asserts:
  - the wake word fires
  - the wake word's tail is NOT returned as a command
  - the real command IS captured within the window
  - a command that starts AFTER the 1 s window is discarded (back to standby)
  - noise alone never fires the wake word
"""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcm_pi_v3 as v3          # noqa: E402

CUE_S = 0.67                    # length of 00_yes.wav


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


def feed(gate, stream, sr, window=1.0):
    """Feed a 48k stream through the gate in 30 ms frames, simulating the
    main thread (cue playback + cue_done). Returns (events, gate)."""
    frame = int(sr * 0.030)
    events = []
    t0 = time.time()
    i = 0
    while i + frame <= len(stream):
        f = stream[i:i + frame]
        t = t0 + i / sr
        ev = gate.push(f, t)
        if ev == "wake":
            events.append((i / sr, "wake"))
            # main thread: mute for the cue, then arm the window
            i += int(sr * CUE_S)
            t = t0 + i / sr
            gate.cue_done(now=t)
        elif ev == "command":
            events.append((i / sr, "command"))
        i += frame
    return events, gate


def make_gate(sr, window=1.0):
    return v3.WakeGate(sr, v3.WakeWord(v3.WAKEWORD_MODEL, threshold=0.5),
                       v3.VAD(sr, gate=0.05), window=window)


def main():
    sr = 48000
    wake_word = synth("hey rhasspy", sr)
    cmd = synth("pause the music", sr)

    # --- case 1: wake -> tail -> command, all within the window ----------
    stream = np.concatenate([
        noise(sr, 1.0),
        wake_word,
        np.zeros(int(sr * 0.3), np.float32),
        cmd,
        noise(sr, 1.0),
    ])
    events, gate = feed(make_gate(sr), stream, sr)
    print("case 1 events:", [(round(t, 2), e) for t, e in events])
    wake_ts = [t for t, e in events if e == "wake"]
    cmd_ts = [t for t, e in events if e == "command"]
    assert wake_ts, "wake word did NOT fire"
    assert cmd_ts, "real command was NOT captured"
    gap = cmd_ts[0] - wake_ts[0]
    assert gap > 0.8, f"command at {gap:.2f}s after wake -- that is the tail"
    a = gate.last_audio
    print(f"  wake at {wake_ts[0]:.2f}s, command at {cmd_ts[0]:.2f}s "
          f"({gap:.2f}s later), captured {len(a)/16000:.2f}s  OK")

    # --- case 2: command starts AFTER the 1 s window -> discarded --------
    stream = np.concatenate([
        noise(sr, 1.0),
        wake_word,
        np.zeros(int(sr * (CUE_S + 1.2))),   # silence past the window
        cmd,
        noise(sr, 1.0),
    ])
    events, gate = feed(make_gate(sr), stream, sr)
    print("case 2 events:", [(round(t, 2), e) for t, e in events])
    assert [e for _, e in events] == ["wake"], \
        f"late command must be discarded, got {events}"
    assert gate.state == gate.WAKE, "gate must be back in WAKE (standby)"
    print("  late command discarded, gate back in standby  OK")

    # --- case 3: noise alone never fires ----------------------------------
    events, gate = feed(make_gate(sr), noise(sr, 5.0, seed=1), sr)
    assert not events, f"noise fired the wake word: {events}"
    print("  5s of noise -> no events  OK")

    # --- case 4: wake, then pure noise in the window -> standby ----------
    stream = np.concatenate([
        noise(sr, 0.5),
        wake_word,
        noise(sr, 3.0, seed=2),
    ])
    events, gate = feed(make_gate(sr), stream, sr)
    print("case 4 events:", [(round(t, 2), e) for t, e in events])
    assert [e for _, e in events] == ["wake"], \
        f"noise in the window must not become a command, got {events}"
    assert gate.state == gate.WAKE
    print("  wake + noise-only window -> back to standby  OK")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
