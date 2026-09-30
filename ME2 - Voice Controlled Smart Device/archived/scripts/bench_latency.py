#!/usr/bin/env python3
"""Inference latency benchmark for the VCM ONNX model (CPU, single thread).

Measures per-utterance latency (feature extraction + ONNX inference +
greedy CTC decode) on a sample of frozen test rows, and reports
p50/p95/p99 latency, throughput, and RTF. This is the number that
transfers to the RPi (same CPU-only ONNX Runtime path).

Usage:
    python3 scripts/bench_latency.py \
        --model model/checkpoints/me2_v5/vcm_int8.onnx \
        --testset data/manifests/frozen_test_v1.jsonl --n 200
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from benchmark import frozen_set  # noqa: E402
from benchmark.models import get_model  # noqa: E402
from benchmark.evaluate import _load_audio  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--testset", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    # Same loader as the benchmark harness (audio paths made absolute).
    rows, _ = frozen_set.load(
        args.testset,
        repo_root=Path(args.testset).resolve().parents[1],
        check_audio=False)
    rng = np.random.default_rng(args.seed)
    rows = [rows[i] for i in rng.choice(len(rows), min(args.n, len(rows)),
                                        replace=False)]

    model = get_model("onnx", args.model)
    model.load()

    # warm-up (JIT, allocator)
    for row in rows[:5]:
        audio, sr = _load_audio(row["audio"])
        model.transcribe(audio, sr=sr)

    lat, dur = [], []
    for row in rows:
        t0 = time.perf_counter()
        audio, sr = _load_audio(row["audio"])
        model.transcribe(audio, sr=sr)
        lat.append(time.perf_counter() - t0)
        dur.append(len(audio) / 16000.0)

    lat = np.array(lat)
    dur = np.array(dur)
    out = {
        "model": args.model,
        "n": len(rows),
        "latency_ms": {
            "p50": round(float(np.percentile(lat, 50)) * 1000, 1),
            "p95": round(float(np.percentile(lat, 95)) * 1000, 1),
            "p99": round(float(np.percentile(lat, 99)) * 1000, 1),
            "mean": round(float(lat.mean()) * 1000, 1),
        },
        "audio_dur_s": {
            "mean": round(float(dur.mean()), 2),
            "p50": round(float(np.percentile(dur, 50)), 2),
        },
        "rtf": round(float(lat.sum() / dur.sum()), 4),
        "throughput_utt_per_s": round(float(len(rows) / lat.sum()), 1),
    }
    print(json.dumps(out, indent=2))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(out, indent=2))
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
