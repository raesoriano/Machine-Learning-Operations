#!/usr/bin/env python3
"""Evaluate the VCM-v2 backbone pipeline (wav2vec2 ASR -> classifier) on a
test set.

Same metrics and ground-truth convention as the archived CTC eval
(archive/ctc_v8/scripts/eval.py) so the numbers are directly comparable:
    folder name -> INTENT (19 dataset labels)
    file name   -> what was SAID (trailing 1/2/3 = take number)
    (folder, spoken) -> one of the 31 commands | REJECT

Reports command accuracy (31-way), intent accuracy (19-way), ASR WER vs the
spoken phrase (with S/I/D breakdown), blank rate, reject behaviour, per-folder
breakdown, full per-clip detail, and CPU inference latency.

Usage:
    python backbone/scripts/eval_w2v2.py \
        --data ../additional_test_data \
        --report backbone/reports/additional_test_w2v2.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
sys.path.insert(0, _BACK)
sys.path.insert(0, os.path.join(_REPO, "archive", "ctc_v8"))

from vcm2b.asr_w2v2 import ASRWav2Vec2            # noqa: E402
from vcm2.classifier import load_classifier, predict  # noqa: E402
from vcm2.ground_truth import build_ground_truth   # noqa: E402
from vcm2.normalize import normalize               # noqa: E402
from vcm2.pipeline import wer                     # noqa: E402

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


def wer_breakdown(ref, hyp):
    """(wer, subs, ins, dels) via edit distance on word lists."""
    ref = normalize(ref).split()
    hyp = normalize(hyp).split()
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    op = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0=-, 1=sub, 2=del, 3=ins
    for i in range(n + 1):
        d[i, 0] = i
        op[i, 0] = 2
    for j in range(m + 1):
        d[0, j] = j
        op[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j], op[i, j] = d[i - 1, j - 1], 0
                continue
            c_sub = d[i - 1, j - 1] + 1
            c_del = d[i - 1, j] + 1
            c_ins = d[i, j - 1] + 1
            best = min(c_sub, c_del, c_ins)
            d[i, j] = best
            op[i, j] = 1 if best == c_sub else (2 if best == c_del else 3)
    s = ins = de = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i, j]
        if o in (1, 2):
            i -= 1
        if o in (1, 3):
            j -= 1
        if o == 1:
            s += 1
        elif o == 2:
            de += 1
        else:
            ins += 1
    return d[n, m] / max(n, 1), s, ins, de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--art", default=None,
                    help="fine-tuned model dir (default backbone/artifacts/w2v2_base)")
    args = ap.parse_args()

    asr = ASRWav2Vec2(art_dir=args.art, device="cpu")
    clf = load_classifier(os.path.join(_REPO, "archive", "ctc_v8",
                                       "artifacts", "classifier.pkl"))

    rows = build_ground_truth(args.data)
    # warm-up (first torch call is slow)
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
        w, s, ins, de = wer_breakdown(r["spoken"], transcript)
        results.append({
            "file": os.path.basename(r["path"]),
            "folder": r["folder"],
            "spoken": r["spoken"],
            "gold": r["gold"],
            "gold_intent": _INTENT_OF.get(r["gold"], "unknown"),
            "transcript": transcript,
            "pred": command,
            "pred_intent": _INTENT_OF.get(command, "unknown"),
            "prob": round(prob, 4),
            "wer": round(w, 3),
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
    w = [r["wer"] for r in results]
    subs = sum(wer_breakdown(r["spoken"], r["transcript"])[1] for r in results)
    ins = sum(wer_breakdown(r["spoken"], r["transcript"])[2] for r in results)
    dels = sum(wer_breakdown(r["spoken"], r["transcript"])[3] for r in results)
    ref_words = sum(len(normalize(r["spoken"]).split()) for r in results)

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
        "model": "vcm-v2 backbone (wav2vec2-base-960h fine-tuned ASR + "
                 "TF-IDF/LogReg classifier)",
        "testset": os.path.basename(os.path.normpath(args.data)),
        "n_clips": n,
        "command_accuracy": round(cmd_correct / n, 4),
        "intent_accuracy_19way": round(intent_correct / n, 4),
        "asr_wer_vs_spoken": round(float(np.mean(w)), 4),
        "wer_edit_counts": {"ref_words": ref_words, "subs": subs,
                            "ins": ins, "dels": dels},
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

    print("\n" + "=" * 66)
    print(f"  VCM v2 BACKBONE on {summary['testset']}  (n={n})")
    print("=" * 66)
    print(f"  command accuracy (31-way) : {100 * summary['command_accuracy']:5.1f}%")
    print(f"  intent accuracy (19-way)  : {100 * summary['intent_accuracy_19way']:5.1f}%")
    print(f"  ASR WER (vs spoken)       : {100 * summary['asr_wer_vs_spoken']:5.1f}%")
    ec = summary["wer_edit_counts"]
    print(f"  WER edits                 : {ec['subs']} subs / {ec['ins']} ins / "
          f"{ec['dels']} dels over {ec['ref_words']} ref words")
    print(f"  blank transcript rate     : {100 * summary['blank_transcript_rate']:5.1f}%")
    print(f"  REJECT emitted / gold     : {reject_pred} / {gold_reject}")
    lt = summary["latency_total_ms"]
    print(f"  latency total ms          : p50={lt['p50']} p95={lt['p95']} p99={lt['p99']}")
    print(f"  latency ASR ms            : p50={summary['latency_asr_ms']['p50']} "
          f"p95={summary['latency_asr_ms']['p95']}")
    print("\n  per-folder (cmd_acc / intent_acc / blank):")
    for f, st in per_folder.items():
        print(f"    {f:18s} {st['cmd_acc'] * 100:5.0f}% / "
              f"{st['intent_acc'] * 100:5.0f}% / {st['blank']}")


if __name__ == "__main__":
    main()
