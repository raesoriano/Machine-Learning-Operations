"""Characterize the noise in additional_test_data vs training audio.

Per-clip: noise-floor level (dBFS) from the quietest 20% of frames, spectral
flatness of the noise floor (1.0 = white, low = tonal/hum), and the dominant
noise-band frequencies.
"""
import glob
import os
import random
from collections import Counter

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_SANDBOX = os.path.dirname(os.path.dirname(_HERE))
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")


def noise_profile(path, sr=16000):
    x, _ = sf.read(path)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = x.astype(np.float64)
    if np.abs(x).max() > 1.5:
        x = x / 32768.0
    n = len(x)
    fr = int(0.025 * sr)
    if n < 6 * fr:
        return None
    starts = list(range(0, n - fr, fr))
    powers = np.array([np.mean(x[i:i + fr] ** 2) for i in starts])
    q = max(2, int(0.2 * len(starts)))
    quiet_idx = np.argsort(powers)[:q]
    quiet = np.concatenate([x[starts[i]:starts[i] + fr] for i in quiet_idx])
    spec = np.abs(np.fft.rfft(quiet * np.hanning(len(quiet)))) ** 2
    freqs = np.fft.rfftfreq(len(quiet), 1 / sr)
    spec = spec + 1e-12
    flat = np.exp(np.mean(np.log(spec))) / np.mean(spec)
    floor_dbfs = 10 * np.log10(np.mean(quiet ** 2) + 1e-12)
    order = np.argsort(spec)[::-1][:3]
    peaks = [(round(float(freqs[i]), 1),
              round(float(spec[i] / np.mean(spec)), 1)) for i in sorted(order)]
    return dict(floor_dbfs=floor_dbfs, flat=flat, peaks=peaks)


def main():
    print("=== additional_test_data (171 clips) ===")
    paths = sorted(glob.glob(os.path.join(_SANDBOX, "additional_test_data",
                                          "*", "*.wav")))
    profs = [p for p in (noise_profile(x) for x in paths) if p]
    floors = [p["floor_dbfs"] for p in profs]
    flats = [p["flat"] for p in profs]
    print(f"  noise-floor level: min {min(floors):.1f} med {np.median(floors):.1f} "
          f"max {max(floors):.1f} dBFS")
    print(f"  spectral flatness: med {np.median(flats):.3f} "
          f"(1.0=white noise, <0.1=tonal/hum)")
    c = Counter()
    for p in profs:
        if p["peaks"]:
            c[round(p["peaks"][0][0] / 100) * 100] += 1
    print(f"  dominant noise-peak freq (Hz, top bin): {dict(sorted(c.items()))}")

    print("\n=== optionb training audio (300 random) ===")
    opt = sorted(glob.glob(os.path.join(_ME2, "data", "optionb", "**", "*.wav"),
                           recursive=True))
    random.seed(0)
    samp = random.sample(opt, min(300, len(opt)))
    profs2 = [p for p in (noise_profile(x) for x in samp) if p]
    floors2 = [p["floor_dbfs"] for p in profs2]
    flats2 = [p["flat"] for p in profs2]
    print(f"  noise-floor level: min {min(floors2):.1f} med {np.median(floors2):.1f} "
          f"max {max(floors2):.1f} dBFS")
    print(f"  spectral flatness: med {np.median(flats2):.3f}")


if __name__ == "__main__":
    main()
