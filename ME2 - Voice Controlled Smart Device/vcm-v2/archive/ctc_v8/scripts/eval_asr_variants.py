#!/usr/bin/env python3
"""Evaluate VCM v2 end-to-end with a CHOSEN ME2 ASR checkpoint.

The production pipeline (vcm2/asr.py) hardwires me2_v6/best.pt. This script
re-runs the same 171-clip `additional_test_data` eval with an explicit
checkpoint so we can compare ASR variants (me2_v6 vs me2_v7, ...) through the
IDENTICAL Stage-2 classifier:

    python scripts/eval_asr_variants.py \
        --ckpt ../Machine-Learning-Operations/ME2\\ -\\ Voice\\ Controlled\\ Smart\\ Device/model/checkpoints/me2_v6/best.pt \
        --tag v6 --report reports/asr_variant_v6.json

Metrics are the same as scripts/eval.py (command acc, 19-way intent acc,
WER vs the spoken phrase, blank rate, reject behaviour, latency).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcm2.asr import ASR                      # noqa: E402
from vcm2.classifier import load_classifier, predict  # noqa: E402
from vcm2.ground_truth import build_ground_truth     # noqa: E402
from vcm2.normalize import normalize                # noqa: E402
from vcm2.pipeline import wer                       # noqa: E402

# 19-way intent labels (same mapping as scripts/eval.py)
_INTENT_OF = {
    "PLAY_MUSIC": "play_music", "WEATHER": "ask_question", "TIME": "ask_question",
    "LIGHT_ON": "lights_switch", "LIGHT_OFF": "lights_switch",
    "BRIGHTNESS_20": "lights_adjust", "BRIGHTNESS_60": "lights_adjust",
    "BRIGHTNESS_100": "lights_adjust",
    "COLOR_RED": "lights_adjust", "COLOR_GREEN": "lights_adjust",
    "COLOR_BLUE": "lights_adjust",
    "TIMER_10s": "set_timer", "TIMER_30s": "set_timer", "TIMER_1m": "set_timer",
    "ALARM_6_00AM": "set_alarm", "ALARM_8_00AM": "set_alarm",
    "ALARM_9_00PM": "set_alarm",
    "TEMPERATURE_18": "set_temperature", "TEMPERATURE_22": "set_temperature",
    "TEMPERATURE_26": "set_temperature",
    "PAUSE": "media_control", "STOP": "media_control", "NEXT": "media_control",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIST_REMINDERS": "reminders_lists",
    "CREATE_REMINDER_DRINK_WATER": "reminders_lists",
    "CREATE_REMINDER_EXERCISE": "reminders_lists",
    "CREATE_REMINDER_STUDY": "reminders_lists",
    "CALL": "call", "MESSAGE": "call", "REJECT": "unknown",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to ME2 best.pt")
    ap.add_argument("--tag", required=True, help="label for the report")
    ap.add_argument("--data", default=None,
                    help="test dir (default: ../additional_test_data)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    data = args.data or os.path.normpath(os.path.join(here, "..", "..",
                                                      "additional_test_data"))
    report = args.report or os.path.join(here, "..", "reports",
                                         f"asr_variant_{args.tag}.json")

    asr = ASR(ckpt=args.ckpt, device="cpu")
    clf = load_classifier()

    rows = build_ground_truth(data)
    a, sr = sf.read(rows[0]["path"], dtype="float32")
    _ = asr.transcribe(a, sr=sr)  # warm-up

    results = []
    t0 = time.perf_counter()
    for r in rows:
        audio, sr = sf.read(r["path"], dtype="float32")
        ta = time.perf_counter()
        transcript = asr.transcribe(audio, sr=sr)
        t_asr = time.perf_counter() - ta
        tc = time.perf_counter()
        command, prob = predict(clf, transcript)
        t_cls = time.perf_counter() - tc
        results.append({
            "file": os.path.basename(r["path"]),
            "folder": r["folder"],
            "spoken": r["spoken"],
            "gold": r["gold"],
            "transcript": transcript,
            "pred": command,
            "prob": round(prob, 4),
            "asr_ms": round(t_asr * 1e3, 1),
            "cls_ms": round(t_cls * 1e3, 2),
        })
    wall = time.perf_counter() - t0

    n = len(results)
    cmd_correct = sum(1 for r in results if r["pred"] == r["gold"])
    intent_correct = sum(
        1 for r in results
        if _INTENT_OF.get(r["pred"], "unknown") == _INTENT_OF.get(r["gold"], "unknown"))
    blank = sum(1 for r in results if not r["transcript"].strip())
    reject_pred = sum(1 for r in results if r["pred"] == "REJECT")
    w = [wer(r["spoken"], r["transcript"]) for r in results]

    per_folder = {}
    for f in sorted({r["folder"] for r in results}):
        rs = [r for r in results if r["folder"] == f]
        per_folder[f] = {
            "n": len(rs),
            "cmd_acc": round(sum(1 for r in rs if r["pred"] == r["gold"]) / len(rs), 3),
            "intent_acc": round(sum(
                1 for r in rs
                if _INTENT_OF.get(r["pred"], "unknown") == _INTENT_OF.get(r["gold"], "unknown")
            ) / len(rs), 3),
            "blank": sum(1 for r in rs if not r["transcript"].strip()),
        }

    lat = [r["asr_ms"] + r["cls_ms"] for r in results]
    summary = {
        "asr_ckpt": os.path.normpath(args.ckpt),
        "tag": args.tag,
        "testset": os.path.basename(os.path.normpath(data)),
        "n_clips": n,
        "command_accuracy": round(cmd_correct / n, 4),
        "intent_accuracy_19way": round(intent_correct / n, 4),
        "asr_wer_vs_spoken": round(float(np.mean(w)), 4),
        "blank_transcript_rate": round(blank / n, 4),
        "reject_emitted": reject_pred,
        "latency_total_ms": {
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
            "p99": round(float(np.percentile(lat, 99)), 1),
            "max": round(float(max(lat)), 1),
        },
        "wall_time_s": round(wall, 1),
    }
    out = {"summary": summary, "per_folder": per_folder, "clips": results}
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"report -> {report}")


if __name__ == "__main__":
    main()
