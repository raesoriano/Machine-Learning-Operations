"""Benchmark harness — one command to validate any VCM on any test set.

Swappable axes (no code changes when either changes):
    --testset synthetic|frozen      (frozen = --testset-manifest path)
    --model     mock_clean|mock_corrupt|onnx|vosk   (--model-path for real ones)

Pipeline per row:  audio -> model.transcribe() -> vcm.parser.parse() -> metrics
This is the SAME pipeline as the RPi runtime, so numbers transfer.

Examples:
    # Harness sanity check (no audio, no model, runs anywhere):
    python -m benchmark.evaluate --testset synthetic --model mock_clean

    # Simulated imperfect ASR front-end:
    python -m benchmark.evaluate --testset synthetic --model mock_corrupt --wer 0.15

    # Your real VCM on the frozen v1 set:
    python -m benchmark.evaluate --testset frozen \
        --testset-manifest benchmark/testset/frozen_v1/manifest.jsonl \
        --model onnx --model-path model/checkpoints/vcm_v1.onnx \
        --report reports/vcm_v1.json

Output: JSON report (summary + per-intent + per-subset + failures) and a
human-readable table on stdout.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark import frozen_set  # noqa: E402
from benchmark.metrics import Metrics  # noqa: E402
from benchmark.models import get_model  # noqa: E402
from benchmark.synthetic_set import build as build_synthetic  # noqa: E402
from vcm.parser import parse  # noqa: E402


def _load_audio(path):
    """wav/flac -> (float32 mono @ 16000 Hz, sr)."""
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        g = __import__("math").gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
        sr = 16000
    return x, sr


def run(args):
    # ---- test set --------------------------------------------------------
    if args.testset == "synthetic":
        rows = list(build_synthetic(args.n, args.seed, args.oov_ratio))
        source = f"synthetic(n={args.n},seed={args.seed})"
    else:
        rows, warnings = frozen_set.load(args.testset_manifest,
                                         repo_root=Path(args.testset_manifest).resolve().parents[1],
                                         check_audio=not args.no_audio_check)
        for w in warnings[:10]:
            print(f"  [warn] {w}", file=sys.stderr)
        source = args.testset_manifest

    # ---- model -----------------------------------------------------------
    model = get_model(args.model, args.model_path,
                      wer=args.wer, seed=args.seed,
                      reject_min_conf=args.reject_min_conf)
    model.load(args.model_path)

    # ---- evaluate --------------------------------------------------------
    m = Metrics()
    failures = []
    t0 = time.time()
    for i, row in enumerate(rows):
        if args.testset == "synthetic":
            # side channel: mock models read the gold text (no audio yet)
            if hasattr(model, "gold"):
                model.gold["_last_"] = row["text"]
            transcript = model.transcribe(None)
        else:
            audio, sr = _load_audio(row["audio"])
            transcript = model.transcribe(audio, sr=sr)
        pred = parse(transcript)
        m.add(row["id"], {"intent": row["intent"], "slots": row["slots"]},
              pred.to_dict(), transcript, row["text"], row.get("subset", "core"))
        gold_ok = row["intent"] == pred.intent
        if not gold_ok or (row["intent"] != "unknown"
                           and row["slots"] != pred.slots):
            failures.append({
                "id": row["id"], "subset": row.get("subset", "core"),
                "gold_text": row["text"], "transcript": transcript,
                "gold": {"intent": row["intent"], "slots": row["slots"]},
                "pred": pred.to_dict(),
            })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(rows)} ...", file=sys.stderr)
    wall = time.time() - t0

    summary = m.summary()
    summary["model"] = model.name
    summary["testset"] = source
    summary["wall_time_s"] = round(wall, 1)
    summary["rtf_estimate"] = round(wall / max(len(rows), 1), 4)

    report = {
        "summary": summary,
        "per_intent": m.per_intent(),
        "per_subset": m.per_subset(),
        "n_failures": len(failures),
        "failures_sample": failures[:50],
    }
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport -> {out}")
    _print_table(summary, m.per_intent(), m.per_subset())
    return report


def _print_table(summary, per_intent, per_subset):
    print("\n" + "=" * 62)
    print(f"  MODEL: {summary['model']}    TESTSET: {summary['testset']}")
    print("=" * 62)
    if "intent_accuracy" in summary:
        print(f"  in-vocab  n={summary['n_in_vocab']:>5}   "
              f"intent acc {summary['intent_accuracy'] * 100:6.2f}%   "
              f"slot F1 {summary['slot_f1'] * 100:6.2f}%   "
              f"exact {summary['exact_match'] * 100:6.2f}%   "
              f"WER {summary['wer'] * 100:5.2f}%")
    if "rejection_rate" in summary:
        print(f"  oov       n={summary['n_oov']:>5}   "
              f"rejection {summary['rejection_rate'] * 100:6.2f}%   "
              f"false-accept {summary['false_accept_rate'] * 100:6.2f}%")
    if "command_accuracy" in summary:
        print(f"  overall   command accuracy {summary['command_accuracy'] * 100:6.2f}%")
    print("-" * 62)
    print("  per-intent (exact / slot F1):")
    for intent, s in per_intent.items():
        print(f"    {intent:<18} {s['exact_match'] * 100:6.2f}% / "
              f"{s['slot_f1'] * 100:6.2f}%   (n={s['n']})")
    print("-" * 62)
    print("  per-subset:")
    for subset, s in per_subset.items():
        bits = []
        if "exact_match" in s:
            bits.append(f"exact {s['exact_match'] * 100:.1f}%")
        if "rejection_rate" in s:
            bits.append(f"rejection {s['rejection_rate'] * 100:.1f}%")
        print(f"    {subset:<10} {'  '.join(bits)}   (n={sum(1 for _ in []) or s.get('n_in_vocab', s.get('n_oov', '?'))})")
    print("=" * 62)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", choices=["synthetic", "frozen"],
                    default="synthetic")
    ap.add_argument("--testset-manifest", default=None,
                    help="JSONL manifest for --testset frozen")
    ap.add_argument("--n", type=int, default=2000, help="synthetic size")
    ap.add_argument("--oov-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=999)
    ap.add_argument("--model", default="mock_clean",
                    choices=["mock_clean", "mock_corrupt", "onnx", "torch", "vosk"])
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--wer", type=float, default=0.15,
                    help="mock_corrupt corruption level")
    ap.add_argument("--report", default=None, help="write JSON report here")
    ap.add_argument("--reject-min-conf", type=float, default=None,
                    help="confidence gate: reject (decode to '') utterances "
                         "whose mean non-blank log-prob is below this "
                         "(tune with scripts/tune_reject_threshold.py)")
    ap.add_argument("--no-audio-check", action="store_true")
    args = ap.parse_args()
    if args.testset == "frozen" and not args.testset_manifest:
        ap.error("--testset frozen requires --testset-manifest")
    if args.model in ("onnx", "torch", "vosk") and not args.model_path:
        ap.error(f"--model {args.model} requires --model-path")
    run(args)


if __name__ == "__main__":
    main()
