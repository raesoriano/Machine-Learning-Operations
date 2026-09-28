"""A/B: does pre-processing (noise removal) improve end-to-end accuracy?

Runs the same 171 additional_test_data clips through the SAME ASR + classifier
under three conditions and compares:
  raw        - unmodified audio (baseline, matches current v2 numbers)
  hp80       - 4th-order Butterworth high-pass at 80 Hz (kills sub-80 Hz rumble)
  nr         - noisereduce stationary STFT spectral subtraction (prop_decrease=0.85)
  hp_nr      - high-pass then noisereduce (cascade)

Reports per condition: blank-transcript count, WER, 31-way command accuracy,
19-way intent accuracy, and mean ASR latency.
"""
import os
import sys
import time
from collections import Counter

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SANDBOX = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import noisereduce as nr  # noqa: E402

from vcm2.asr import ASR  # noqa: E402
from vcm2.classifier import load_classifier, predict  # noqa: E402
from vcm2.ground_truth import build_ground_truth  # noqa: E402
from vcm2.pipeline import wer  # noqa: E402
from vcm2.normalize import normalize  # noqa: E402

DATA = os.path.join(_SANDBOX, "additional_test_data")

# 19-way intent = first token of the command (ALARM_6_00AM -> ALARM)
def intent_of(cmd):
    return cmd.split("_")[0] if cmd != "REJECT" else "REJECT"


def hp80(x, sr=16000):
    sos = butter(4, 80, btype="highpass", fs=sr, output="sos")
    return sosfilt(sos, x)


def denoise(x, sr=16000, prop=0.85):
    return nr.reduce_noise(y=x, sr=sr, stationary=True,
                           prop_decrease=prop,
                           n_fft=512, hop_length=128)


def main():
    rows = build_ground_truth(DATA)
    asr = ASR()
    clf = load_classifier()
    print(f"loaded ASR + classifier; {len(rows)} clips\n")

    # warm up
    x, sr = sf.read(rows[0]["path"])
    asr.transcribe(x, sr)

    conds = ["raw", "hp80", "nr", "hp_nr"]
    acc = {c: [] for c in conds}
    intent_acc = {c: [] for c in conds}
    blanks = {c: 0 for c in conds}
    wers = {c: [] for c in conds}
    lat = {c: [] for c in conds}
    details = []

    for r in rows:
        x, sr = sf.read(r["path"])
        if x.ndim > 1:
            x = x.mean(axis=1)
        x = x.astype(np.float32)
        if np.abs(x).max() > 1.5:
            x = x / 32768.0
        variants = {
            "raw": x,
            "hp80": hp80(x).astype(np.float32),
            "nr": denoise(x).astype(np.float32),
            "hp_nr": hp80(denoise(x)).astype(np.float32),
        }
        row = {"folder": r["folder"], "spoken": r["spoken"], "gold": r["gold"]}
        for c in conds:
            t0 = time.perf_counter()
            tr = asr.transcribe(variants[c], sr)
            lat[c].append((time.perf_counter() - t0) * 1e3)
            cmd, _ = predict(clf, tr)
            acc[c].append(cmd == r["gold"])
            intent_acc[c].append(intent_of(cmd) == intent_of(r["gold"]))
            if not tr.strip():
                blanks[c] += 1
            wers[c].append(wer(r["spoken"], tr))
            row[c] = (tr, cmd)
        details.append(row)

    print(f"{'cond':8s} {'blank':>5s} {'WER':>6s} {'cmd%':>6s} {'intent%':>8s} "
          f"{'asr_ms':>7s}")
    for c in conds:
        print(f"{c:8s} {blanks[c]:5d} {np.mean(wers[c])*100:6.1f} "
              f"{np.mean(acc[c])*100:6.1f} {np.mean(intent_acc[c])*100:8.1f} "
              f"{np.mean(lat[c]):7.1f}")

    # per-clip diff: raw vs best enhancer
    print("\nclips FIXED by enhancement (raw wrong -> enhancer right):")
    fixed = [d for d in details
             if d["raw"][1] != d["gold"] and d["nr"][1] == d["gold"]]
    broken = [d for d in details
              if d["raw"][1] == d["gold"] and d["nr"][1] != d["gold"]]
    for d in fixed[:25]:
        print(f"  {d['folder']:16s} said {d['spoken']!r:34s} "
              f"raw {d['raw'][0]!r:20s} -> {d['raw'][1]:22s} | "
              f"nr {d['nr'][0]!r:20s} -> {d['nr'][1]}")
    print(f"\nclips BROKEN by enhancement (raw right -> enhancer wrong): {len(broken)}")
    for d in broken[:10]:
        print(f"  {d['folder']:16s} said {d['spoken']!r:34s} "
              f"raw {d['raw'][0]!r:20s} -> {d['raw'][1]:22s} | "
              f"nr {d['nr'][0]!r:20s} -> {d['nr'][1]}")

    # save full per-clip results
    import json
    out = {
        "summary": {c: {
            "blank": blanks[c],
            "wer": round(float(np.mean(wers[c])), 4),
            "cmd_acc": round(float(np.mean(acc[c])), 4),
            "intent_acc": round(float(np.mean(intent_acc[c])), 4),
            "asr_ms_mean": round(float(np.mean(lat[c])), 2),
        } for c in conds},
        "clips": [{
            "folder": d["folder"], "spoken": d["spoken"], "gold": d["gold"],
            **{c: {"transcript": d[c][0], "command": d[c][1]} for c in conds},
        } for d in details],
    }
    rep = os.path.join(_ROOT, "reports", "noise_ab.json")
    with open(rep, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {rep}")


if __name__ == "__main__":
    main()
