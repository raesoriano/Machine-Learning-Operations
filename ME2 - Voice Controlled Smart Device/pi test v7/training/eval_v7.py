#!/usr/bin/env python3
"""pi test v7 -- evaluate the PocketSphinx ensemble (v3 architecture) with
the custom AM retrained on the v6 dataset, on the v6 test split.

Same protocol as v3's eval:
  * grammar-constrained decode (103-phrase JSGF) with the custom AM
  * + the stock en-us AM, same grammar
  * stage-2 TF-IDF/LR classifier (v3's classifier.pkl) maps the transcript
    to one of the 31 fine commands + REJECT
  * ensemble fusion: agreement, else the more confident classifier
  * gold: manifest `command` column (coarse 19 schema) via the fine->coarse
    map; OUT_OF_SCOPE rows are the REJECT class
  * report: overall / command / reject / intent accuracy, real-vs-synthetic
    split, per-class, latency

Run:
    python training/eval_v7.py --split test --report _eval_test.json
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)          # pi test v7/
_ME2 = os.path.dirname(_BACK)
_V6 = os.path.join(_ME2, "pi test v6")  # hgm package (maps + tokenizer)
sys.path.insert(0, _BACK)               # vcm2 package
sys.path.insert(0, _V6)                 # hgm package for the maps

from vcm2.classifier import load_classifier, predict      # noqa: E402
from hgm.commands import coarse_class                     # noqa: E402

CUSTOM_HMM = os.path.join(_BACK, "am", "custom")
CUSTOM_LM = os.path.join(CUSTOM_HMM, "vcm.lm.bin")
STOCK_HMM = os.path.join(_BACK, "am", "stock")
STOCK_DICT = os.path.join(_BACK, "am", "stock_enus",
                          "cmudict-en-us.dict")
STOCK_LM = os.path.join(_BACK, "am", "stock_enus", "en-us.lm.bin")
DICT3 = os.path.join(_BACK, "dict3")
JSGF = os.path.join(_BACK, "vcm_commands_enh3.jsgf")
CLASSIFIER = os.path.join(_BACK, "classifier.pkl")

DEFAULT_DATA = "/home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset"

# fine 31-command -> coarse 19 schema (v3 ontology -> v6 schema)
FINE_TO_COARSE = {
    "ALARM_6_00AM": "ALARM", "ALARM_8_00AM": "ALARM", "ALARM_9_00PM": "ALARM",
    "BRIGHTNESS_100": "BRIGHTNESS", "BRIGHTNESS_60": "BRIGHTNESS",
    "BRIGHTNESS_20": "BRIGHTNESS",
    "COLOR_RED": "COLOR", "COLOR_GREEN": "COLOR", "COLOR_BLUE": "COLOR",
    "TEMPERATURE_18": "TEMPERATURE", "TEMPERATURE_22": "TEMPERATURE",
    "TEMPERATURE_26": "TEMPERATURE",
    "CREATE_REMINDER_DRINK_WATER": "CREATE_REMINDER",
    "CREATE_REMINDER_EXERCISE": "CREATE_REMINDER",
    "CREATE_REMINDER_STUDY": "CREATE_REMINDER",
    "LIST_REMINDERS": "LIST_REMINDERS",
    "TIMER_10s": "TIMER", "TIMER_30s": "TIMER", "TIMER_1m": "TIMER",
    "VOLUME_UP": "VOLUME_UP", "VOLUME_DOWN": "VOLUME_DOWN",
    "LIGHT_OFF": "LIGHT_OFF", "LIGHT_ON": "LIGHT_ON",
    "PLAY_MUSIC": "PLAY_MUSIC", "NEXT": "NEXT", "PAUSE": "PAUSE",
    "STOP": "STOP", "TIME": "TIME", "WEATHER": "WEATHER",
    "CALL": "CALL", "MESSAGE": "MESSAGE",
    "REJECT": "REJECT",
}

COARSE_TO_INTENT = {
    "ALARM": "set_alarm",
    "BRIGHTNESS": "lights_adjust", "COLOR": "lights_adjust",
    "TEMPERATURE": "set_temperature",
    "CREATE_REMINDER": "reminders_lists", "LIST_REMINDERS": "reminders_lists",
    "TIMER": "set_timer",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIGHT_OFF": "lights_switch", "LIGHT_ON": "lights_switch",
    "PLAY_MUSIC": "play_music", "NEXT": "media_control",
    "PAUSE": "media_control", "STOP": "media_control",
    "TIME": "ask_question", "WEATHER": "ask_question",
    "CALL": "call", "MESSAGE": "call",
}

_CLF = None
_DEC_C = None
_DEC_S = None


def _init_worker():
    global _CLF, _DEC_C, _DEC_S
    from pocketsphinx import Config, Decoder

    def _dec(hmm, dictionary, lm):
        cfg = Config()
        cfg.set_string("-hmm", hmm)
        cfg.set_string("-dict", dictionary)
        cfg.set_string("-lm", lm)
        cfg.set_string("-logfn", "/dev/null")
        cfg.set_string("-wip", "0.65")
        d = Decoder(cfg)
        with open(JSGF) as f:
            d.add_jsgf_string("vcm", f.read())
        d.activate_search("vcm")
        return d

    _DEC_C = _dec(CUSTOM_HMM, DICT3, CUSTOM_LM)
    _DEC_S = _dec(STOCK_HMM, STOCK_DICT, STOCK_LM)
    _CLF = load_classifier(CLASSIFIER)


def _eval_one(job):
    path, gold, is_syn = job
    t0 = time.perf_counter()
    texts = []
    for dec in (_DEC_C, _DEC_S):
        dec.start_utt()
        with wave.open(path, "rb") as w:
            while True:
                frames = w.readframes(4000)
                if not frames:
                    break
                dec.process_raw(frames, False, False)
        dec.end_utt()
        hyp = dec.hyp()
        texts.append((hyp.hypstr or "").strip() if hyp else "")
    ms = (time.perf_counter() - t0) * 1000.0
    cc, cpc = predict(_CLF, texts[0])
    cs, cps = predict(_CLF, texts[1])
    if cc == cs:
        cmd, transcript = cc, texts[0]
    elif cpc >= cps:
        cmd, transcript = cc, texts[0]
    else:
        cmd, transcript = cs, texts[1]
    pred = FINE_TO_COARSE.get(cmd, "REJECT")
    gold_c = "REJECT" if gold == "OUT_OF_SCOPE" else gold
    return {
        "file": os.path.basename(path),
        "gold": gold_c,
        "pred": pred,
        "transcript": transcript,
        "correct": pred == gold_c,
        "is_syn": is_syn,
        "asr_ms": round(ms, 1),
    }


def _load_split(data, split):
    rows = []
    mpath = os.path.join(data, split, "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            gold = (r.get("command") or "").strip()
            if not gold:
                continue
            p = os.path.join(data, split, r["file"])
            if os.path.exists(p):
                rows.append((p, gold, int(r.get("is_synthetic") or 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.report is None:
        args.report = os.path.join(_BACK, f"_eval_{args.split}.json")

    rows = _load_split(args.data, args.split)
    print(f"{args.split} clips: {len(rows)}", flush=True)

    results = [None] * len(rows)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(_eval_one, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(rows)}  ({time.time() - t0:.0f}s)",
                      flush=True)

    n = len(results)
    inscope = [r for r in results if r["gold"] != "REJECT"]
    oos = [r for r in results if r["gold"] == "REJECT"]
    overall = sum(r["correct"] for r in results) / max(1, n)
    cmd_acc = sum(r["correct"] for r in inscope) / max(1, len(inscope))
    rej_acc = sum(r["correct"] for r in oos) / max(1, len(oos))
    intent_acc = sum(COARSE_TO_INTENT.get(r["pred"], "unknown") ==
                     COARSE_TO_INTENT.get(r["gold"], "unknown")
                     for r in results) / max(1, n)
    lat = np.array([r["asr_ms"] for r in results])

    per_class = {}
    for c in sorted(set(r["gold"] for r in results)):
        sub = [r for r in results if r["gold"] == c]
        per_class[c] = {"n": len(sub),
                        "acc": round(sum(r["correct"] for r in sub)
                                     / len(sub), 4)}

    def _block(sub):
        if not sub:
            return {"n": 0}
        ins = [r for r in sub if r["gold"] != "REJECT"]
        oos_ = [r for r in sub if r["gold"] == "REJECT"]
        return {
            "n": len(sub),
            "overall_acc": round(sum(r["correct"] for r in sub) / len(sub), 4),
            "command_acc": round(sum(r["correct"] for r in ins) /
                                 max(1, len(ins)), 4),
            "reject_acc": round(sum(r["correct"] for r in oos_) /
                                max(1, len(oos_)), 4),
        }

    report = {
        "model": "v7 PocketSphinx ensemble (v3 architecture, custom AM "
                 "retrained on the v6 dataset)",
        "split": args.split, "n_clips": n,
        "n_inscope": len(inscope), "n_oos": len(oos),
        "overall_acc": round(overall, 4),
        "command_acc": round(cmd_acc, 4),
        "reject_acc": round(rej_acc, 4),
        "intent_acc": round(intent_acc, 4),
        "asr_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "asr_ms_p90": round(float(np.percentile(lat, 90)), 1),
        "real": _block([r for r in results if not r["is_syn"]]),
        "synthetic": _block([r for r in results if r["is_syn"]]),
        "per_class": per_class,
        "results": results,
    }
    json.dump(report, open(args.report, "w"), indent=1)
    print(f"\n=== {args.split} ({n} clips) ===")
    print(f"  overall_acc : {overall:.4f}")
    print(f"  command_acc : {cmd_acc:.4f}  ({len(inscope)} in-scope)")
    print(f"  reject_acc  : {rej_acc:.4f}  ({len(oos)} OOS)")
    print(f"  intent_acc  : {intent_acc:.4f}")
    print(f"  latency ms  : p50={report['asr_ms_p50']}  p90={report['asr_ms_p90']}")
    print(f"  REAL        : n={report['real']['n']}  overall={report['real'].get('overall_acc')}  cmd={report['real'].get('command_acc')}  rej={report['real'].get('reject_acc')}")
    print(f"  SYNTHETIC   : n={report['synthetic']['n']}  overall={report['synthetic'].get('overall_acc')}  cmd={report['synthetic'].get('command_acc')}  rej={report['synthetic'].get('reject_acc')}")
    print(f"  report      : {args.report}")


if __name__ == "__main__":
    main()
