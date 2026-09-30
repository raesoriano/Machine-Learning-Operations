"""RPi simulation — run the FULL on-device VCM pipeline on this x86 machine.

Why this is a valid simulation:
  The RPi runtime (deploy/rpi_service) is:
      VAD -> log_mel (vcm.features) -> ONNX int8 (onnxruntime CPU)
          -> greedy CTC decode -> vcm.parser.parse() -> local backend
  Every stage except microphone capture is pure CPU numpy/onnxruntime with
  NO RPi-specific code. Running it here on x86 exercises the identical
  model, features, decoder, and parser. The only differences on the real
  RPi are (a) the mic driver and (b) CPU speed (Cortex-A72 vs x86).

What this script measures:
  * functional: every test-set row -> transcript -> command (same as RPi)
  * efficiency: per-row wall time, RTF (compute / audio duration), p50/p95
    latency, peak RSS, model file size.

Interpreting the numbers for the RPi:
  * RTF and latency measured here are an UPPER bound on the RPi only if the
    RPi CPU is slower — which it is (Cortex-A72 ~ 4x slower per core than a
    modern x86 core, but the RPi4/5 has 4-8 cores and NEON). Rule of thumb
    for this model size (~1.3M params, int8): if RTF <= 0.25 here, expect
    RTF <= 0.5 on RPi4 (the PLAN.md target). For a definitive sign-off, run
    this same script ON the RPi (it has no RPi-specific deps).

Usage:
    python -m deploy.simulate_rpi \
        --onnx model/checkpoints/vcm_v1_int8.onnx \
        --testset data/manifests/slurp_test.jsonl \
        --report reports/rpi_sim.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from vcm.features import log_mel  # noqa: E402
from vcm.parser import parse  # noqa: E402
from vcm.vocab import ID2WORD  # noqa: E402


def greedy_ctc_to_text(logits, blank=0):
    frame_ids = np.argmax(logits, axis=-1).tolist()
    out, prev = [], None
    for i in frame_ids:
        if i != blank and i != prev:
            out.append(ID2WORD[i - 1])
        prev = i
    return " ".join(out)


def peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--testset", required=True, help="JSONL manifest (frozen schema)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    rows = [json.loads(l) for l in open(args.testset) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    lat, rtf, correct, total, dur_sum = [], [], 0, 0, 0.0
    for row in rows:
        sr, x = wavfile.read(row["audio"])
        x = x.astype(np.float32)
        mel = log_mel(x, sr=sr)
        t0 = time.perf_counter()
        logits = sess.run(None, {in_name: mel[None]})[0][0]
        transcript = greedy_ctc_to_text(logits)
        dt = time.perf_counter() - t0
        lat.append(dt)
        dur = len(x) / sr
        dur_sum += dur
        rtf.append(dt / dur)
        pred = parse(transcript)
        if pred.intent == row["intent"]:
            correct += 1
        total += 1

    lat = np.array(lat)
    rtf = np.array(rtf)
    result = {
        "onnx": str(args.onnx),
        "model_size_mb": round(Path(args.onnx).stat().st_size / 1e6, 3),
        "n_rows": total,
        "intent_accuracy": round(correct / max(total, 1), 4),
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)) * 1000, 1),
            "p95": round(float(np.percentile(lat, 95)) * 1000, 1),
            "max": round(float(lat.max()) * 1000, 1),
        },
        "rtf": {
            "mean": round(float(rtf.mean()), 4),
            "p95": round(float(np.percentile(rtf, 95)), 4),
            "max": round(float(rtf.max()), 4),
        },
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "audio_hours": round(dur_sum / 3600, 4),
        "note": "x86 simulation; RTF/latency here are the best-case. "
                "For RPi4 sign-off run this script on the RPi itself.",
    }
    print(json.dumps(result, indent=2))
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))
        print(f"report -> {out}")


if __name__ == "__main__":
    main()
