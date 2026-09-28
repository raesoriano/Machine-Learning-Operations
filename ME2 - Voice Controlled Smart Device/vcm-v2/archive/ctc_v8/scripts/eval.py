#!/usr/bin/env python3
"""Evaluate VCM v2 (two-stage ASR -> classifier) on a test set.

Ground truth for `additional_test_data`:
    folder name -> INTENT (19 dataset labels)
    file name   -> what was SAID (trailing 1/2/3 = take number)
    (folder, spoken) -> one of the 31 commands | REJECT  (vcm2.ground_truth)

Reports:
    * command accuracy (31-way, the device-level metric)
    * intent accuracy (19-way, for comparison with v1)
    * ASR transcript quality (WER vs the spoken phrase)
    * blank-transcript rate (the v1 failure mode)
    * reject behaviour (how often REJECT is emitted)
    * per-folder breakdown + full per-clip detail
    * inference latency (ASR + classifier, CPU)

    python scripts/eval.py --data ../additional_test_data \
        --report reports/additional_test_v2.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcm2.asr import ASR
from vcm2.classifier import load_classifier, predict
from vcm2.ground_truth import build_ground_truth
from vcm2.normalize import normalize
from vcm2.pipeline import wer

# 19-way intent labels (the space v1 was scored in) for comparison
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
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--ckpt", default=None,
                    help="ASR checkpoint (default: me2_v6). e.g. "
                         "artifacts/asr/me2_v8/best.pt")
    args = ap.parse_args()

    asr = ASR(ckpt=args.ckpt, device="cpu") if args.ckpt else ASR(device="cpu")
    clf = load_classifier()

    rows = build_ground_truth(args.data)
    # warm-up (first ONNX/torch call is slow)
    a, sr = sf.read(rows[0]["path"], dtype="float32")
    _ = asr.transcribe(a, sr=sr)

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
            "gold_intent": _INTENT_OF.get(r["gold"], "unknown"),
            "transcript": transcript,
            "transcript_norm": normalize(transcript),
            "pred": command,
            "pred_intent": _INTENT_OF.get(command, "unknown"),
            "prob": round(prob, 4),
            "asr_ms": round(t_asr * 1e3, 1),
            "cls_ms": round(t_cls * 1e3, 2),
            "total_ms": round((t_asr + t_cls) * 1e3, 1),
        })
    wall = time.perf_counter() - t0

    n = len(results)
    cmd_correct = sum(1 for r in results if r["pred"] == r["gold"])
    intent_correct = sum(1 for r in results if r["pred_intent"] == r["gold_intent"])
    blank = sum(1 for r in results if not r["transcript"].strip())
    reject_pred = sum(1 for r in results if r["pred"] == "REJECT")
    gold_reject = sum(1 for r in results if r["gold"] == "REJECT")
    w = [wer(r["spoken"], r["transcript"]) for r in results]

    per_folder = {}
    for f in sorted({r["folder"] for r in results}):
        rs = [r for r in results if r["folder"] == f]
        per_folder[f] = {
            "n": len(rs),
            "cmd_acc": round(sum(1 for r in rs if r["pred"] == r["gold"]) / len(rs), 3),
            "intent_acc": round(sum(1 for r in rs if r["pred_intent"] == r["gold_intent"]) / len(rs), 3),
            "blank": sum(1 for r in rs if not r["transcript"].strip()),
        }

    lat = [r["total_ms"] for r in results]
    lat_asr = [r["asr_ms"] for r in results]
    lat_cls = [r["cls_ms"] for r in results]

    summary = {
        "model": "vcm-v2 (" + (os.path.basename(os.path.dirname(args.ckpt))
                               if args.ckpt else "me2_v6") +
                 " ASR + TF-IDF/LogReg classifier)",
        "testset": os.path.basename(os.path.normpath(args.data)),
        "n_clips": n,
        "command_accuracy": round(cmd_correct / n, 4),
        "intent_accuracy_19way": round(intent_correct / n, 4),
        "asr_wer_vs_spoken": round(float(np.mean(w)), 4),
        "blank_transcript_rate": round(blank / n, 4),
        "reject_emitted": reject_pred,
        "gold_reject": gold_reject,
        "latency_total_ms": {
            "p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1),
            "p99": round(float(np.percentile(lat, 99)), 1),
            "max": round(float(max(lat)), 1),
        },
        "latency_asr_ms": {"p50": round(float(np.percentile(lat_asr, 50)), 1),
                           "p95": round(float(np.percentile(lat_asr, 95)), 1)},
        "latency_classifier_ms": {"p50": round(float(np.percentile(lat_cls, 50)), 2),
                                  "p95": round(float(np.percentile(lat_cls, 95)), 2)},
        "wall_time_s": round(wall, 1),
    }

    report = {"summary": summary, "per_folder": per_folder, "clips": results}
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nreport -> {args.report}")

    # console summary
    print("\n" + "=" * 66)
    print(f"  VCM v2 on {summary['testset']}  (n={n})")
    print("=" * 66)
    print(f"  command accuracy (31-way) : {100*summary['command_accuracy']:5.1f}%")
    print(f"  intent accuracy (19-way)  : {100*summary['intent_accuracy_19way']:5.1f}%")
    print(f"  ASR WER (vs spoken)       : {100*summary['asr_wer_vs_spoken']:5.1f}%")
    print(f"  blank transcript rate     : {100*summary['blank_transcript_rate']:5.1f}%")
    print(f"  REJECT emitted / gold     : {reject_pred} / {gold_reject}")
    lt = summary["latency_total_ms"]
    print(f"  latency total ms          : p50={lt['p50']} p95={lt['p95']} p99={lt['p99']}")
    print(f"  latency ASR ms            : p50={summary['latency_asr_ms']['p50']} p95={summary['latency_asr_ms']['p95']}")
    print(f"  latency classifier ms     : p50={summary['latency_classifier_ms']['p50']} p95={summary['latency_classifier_ms']['p95']}")
    print("\n  per-folder (cmd_acc / intent_acc / blank):")
    for f, st in per_folder.items():
        print(f"    {f:18s} {st['cmd_acc']*100:5.0f}% / {st['intent_acc']*100:5.0f}% / {st['blank']}")
    return report


if __name__ == "__main__":
    main()
