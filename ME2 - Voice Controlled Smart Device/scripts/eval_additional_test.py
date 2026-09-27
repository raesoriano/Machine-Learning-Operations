"""Evaluate the VCM on the hand-recorded `additional_test_data` set.

Ground-truth convention (per the data owner):
    * folder name  -> the INTENT (one of the 19 dataset labels)
    * file name    -> what was SAID (the transcript); a trailing `1`/`2`/`3`
                      is a take number, not part of the utterance.

The 19 folder labels are mapped onto the parser's 11-intent space (the space
the model is scored in). Gold slots are derived from the transcript with the
SAME parser used to score the model (consistent with the frozen-v1
"parser-canonical" gold). If a transcript is outside the parser grammar
(e.g. "end playback"), gold falls back to the folder's canonical slots.

Runs the int8 ONNX (the exact RPi inference path) and reports the same
metrics as benchmark.evaluate plus per-clip inference latency.

    python scripts/eval_additional_test.py \
        --data ../../additional_test_data \
        --model model/checkpoints/me2_v5/vcm_int8.onnx \
        --report reports/additional_test_int8.json
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.metrics import Metrics  # noqa: E402
from benchmark.models import get_model  # noqa: E402
from vcm.parser import parse  # noqa: E402

# 19 dataset labels -> parser 11-intent space (the space the model emits).
FOLDER_INTENT = {
    "PLAY_MUSIC": "play_music",
    "WEATHER": "ask_question",
    "TIME": "ask_question",
    "LIGHT_ON": "lights_switch",
    "LIGHT_OFF": "lights_switch",
    "PAUSE": "media_control",
    "STOP": "media_control",
    "NEXT": "media_control",
    "VOLUME_UP": "media_control",
    "VOLUME_DOWN": "media_control",
    "CALL": "call",
    "MESSAGE": "call",          # parser folds message into call {action:message}
    "LIST_REMINDERS": "reminders_lists",
    "CREATE_REMINDER": "reminders_lists",
    "TIMER": "set_timer",
    "ALARM": "set_alarm",
    "TEMPERATURE": "set_temperature",
    "BRIGHTNESS": "lights_adjust",
    "COLOR": "lights_adjust",
}

# Canonical (intent, slots) per folder, used only when a transcript is outside
# the parser grammar so gold still reflects the intended command.
FOLDER_CANONICAL = {
    "STOP": {"intent": "media_control", "slots": {"action": "stop"}},
    "VOLUME_UP": {"intent": "media_control", "slots": {"action": "volume_up"}},
}


def gold_text_from(fname: str) -> str:
    """Filename -> transcript. Strip the trailing take number (1..5)."""
    t = os.path.splitext(fname)[0]
    toks = t.split()
    if toks and toks[-1].isdigit() and int(toks[-1]) <= 5:
        toks = toks[:-1]
    return " ".join(toks)


def build_rows(data_dir: str):
    rows = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*", "*.wav"))):
        folder = os.path.basename(os.path.dirname(p))
        if folder not in FOLDER_INTENT:
            raise ValueError(f"unknown intent folder: {folder}")
        gt = gold_text_from(os.path.basename(p))
        parsed = parse(gt)
        if parsed.intent == FOLDER_INTENT[folder]:
            gold = {"intent": FOLDER_INTENT[folder], "slots": parsed.slots}
        else:
            gold = dict(FOLDER_CANONICAL[folder])
        rows.append({
            "id": f"{folder}/{os.path.basename(p)}",
            "audio": p,
            "text": gt,
            "intent": gold["intent"],
            "slots": gold["slots"],
            "folder": folder,
            "subset": "in_domain",
        })
    return rows


def run(args):
    rows = build_rows(args.data)
    if not rows:
        raise SystemExit(f"no wav files found under {args.data}")
    print(f"loaded {len(rows)} rows from {args.data}")

    model = get_model("onnx", args.model)
    model.load(args.model)

    import soundfile as sf
    import numpy as np

    m = Metrics()
    failures = []
    lat = []
    t0 = time.time()
    for i, row in enumerate(rows):
        x, sr = sf.read(row["audio"], dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        ti = time.time()
        transcript = model.transcribe(x, sr=sr)
        lat.append((time.time() - ti) * 1000.0)
        pred = parse(transcript)
        m.add(row["id"], {"intent": row["intent"], "slots": row["slots"]},
              pred.to_dict(), transcript, row["text"], row["subset"])
        ok = (row["intent"] == pred.intent
              and row["slots"] == pred.slots)
        if not ok:
            failures.append({
                "id": row["id"], "folder": row["folder"],
                "gold_text": row["text"], "transcript": transcript,
                "gold": {"intent": row["intent"], "slots": row["slots"]},
                "pred": pred.to_dict(),
            })
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(rows)} ...", file=sys.stderr)
    wall = time.time() - t0

    lat = np.array(lat)
    summary = m.summary()
    summary["model"] = "onnx-int8"
    summary["testset"] = args.data
    summary["wall_time_s"] = round(wall, 1)
    summary["inference_ms"] = {
        "mean": round(float(lat.mean()), 1),
        "p50": round(float(np.percentile(lat, 50)), 1),
        "p95": round(float(np.percentile(lat, 95)), 1),
        "p99": round(float(np.percentile(lat, 99)), 1),
        "max": round(float(lat.max()), 1),
    }

    # per-folder (the user's intent labels) breakdown
    per_folder = {}
    for r in m.rows:
        f = r["id"].split("/")[0]
        per_folder.setdefault(f, []).append(r)
    folder_stats = {}
    for f, rs in sorted(per_folder.items()):
        exact = sum(1 for r in rs
                    if r["pred"]["intent"] == r["gold"]["intent"]
                    and r["gold"]["slots"] == r["pred"]["slots"])
        intent_ok = sum(1 for r in rs
                        if r["pred"]["intent"] == r["gold"]["intent"])
        folder_stats[f] = {
            "n": len(rs),
            "intent_acc": round(intent_ok / len(rs), 3),
            "exact": round(exact / len(rs), 3),
        }

    report = {
        "summary": summary,
        "per_intent": m.per_intent(),
        "per_folder": folder_stats,
        "n_failures": len(failures),
        "failures": failures,
    }
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nreport -> {out}")
    _print(summary, folder_stats)
    return report


def _print(summary, folder_stats):
    print("\n" + "=" * 64)
    print(f"  MODEL: {summary['model']}   TESTSET: {summary['testset']}")
    print("=" * 64)
    print(f"  n_in_vocab   {summary['n_in_vocab']:>5}")
    print(f"  intent acc   {summary['intent_accuracy'] * 100:6.2f}%")
    print(f"  slot F1      {summary['slot_f1'] * 100:6.2f}%")
    print(f"  exact match  {summary['exact_match'] * 100:6.2f}%")
    print(f"  WER          {summary['wer'] * 100:6.2f}%")
    inf = summary["inference_ms"]
    print(f"  infer ms     mean {inf['mean']}  p50 {inf['p50']}  "
          f"p95 {inf['p95']}  p99 {inf['p99']}  max {inf['max']}")
    print("-" * 64)
    print("  per-folder (intent acc / exact):")
    for f, s in folder_stats.items():
        print(f"    {f:<16} {s['intent_acc'] * 100:6.1f}% / "
              f"{s['exact'] * 100:6.1f}%   (n={s['n']})")
    print("=" * 64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../../additional_test_data")
    ap.add_argument("--model",
                    default="model/checkpoints/me2_v5/vcm_int8.onnx")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
