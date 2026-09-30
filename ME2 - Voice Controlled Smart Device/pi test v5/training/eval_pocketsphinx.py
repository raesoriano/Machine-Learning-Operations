#!/usr/bin/env python3
"""Evaluate a PocketSphinx configuration on the 171-clip sole test set
(test_data/additional_test_data) with the SAME ground truth, stage-2
classifier, and WER definition as the Whisper fine-tune eval
(backbone/scripts/eval_whisper_ft.py), so numbers are directly comparable.

Modes
-----
* free dictation : -hmm/-dict/-lm point at an acoustic model + dictionary +
                   bigram LM. The transcript is free text.
* command grammar: additionally -jsgf points at a JSGF file; the decoder is
                   constrained to the 93 command phrases (how PocketSphinx is
                   actually deployed for command recognition).

The transcript (whatever the mode) is then fed to the SAME stage-2
classifier (TF-IDF + LR, 31 commands + REJECT) the Whisper pipeline uses, so
"command_acc" is measured identically across models.

Usage
-----
    python backbone/pocketsphinx/eval_pocketsphinx.py \
        --name stock_free \
        --hmm <...>/model/en-us/en-us --dict <...>/cmudict-en-us.dict \
        --lm <...>/en-us.lm.bin \
        --report backbone/reports/pocketsphinx_stock_free.json

    python backbone/pocketsphinx/eval_pocketsphinx.py \
        --name stock_cmd \
        --hmm <...>/model/en-us/en-us --dict <...>/cmudict-en-us.dict \
        --lm <...>/en-us.lm.bin \
        --jsgf backbone/pocketsphinx/vcm_commands.jsgf \
        --report backbone/reports/pocketsphinx_stock_cmd.json
"""
import argparse
import json
import os
import re
import sys
import time
import wave

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_ME2 = os.path.dirname(_REPO)
sys.path.insert(0, _BACK)   # pi test v5/ (the vcm2 package that ships with the model)
sys.path.insert(0, _HERE)   # training/ (local modules)

from vcm2.classifier import load_classifier, predict      # noqa: E402
from vcm2.ground_truth import build_ground_truth          # noqa: E402
from vcm2.normalize import normalize                      # noqa: E402

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
    "CALL": "call", "MESSAGE": "reminders_lists",
    "REJECT": "reject",
}


def wer_breakdown(ref, hyp):
    """Identical to eval_whisper_ft.py (normalize + edit distance / |ref|)."""
    ref = normalize(ref).split()
    hyp = normalize(hyp).split()[:200]
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n + 1):
        d[i, 0] = i
    for j in range(m + 1):
        d[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
                continue
            d[i, j] = min(d[i - 1, j - 1] + 1, d[i - 1, j] + 1, d[i, j - 1] + 1)
    return d[n, m] / max(n, 1)


def make_decoder(hmm, dictionary, lm, jsgf=None, extra=None):
    from pocketsphinx import Config, Decoder
    cfg = Config()
    cfg.set_string("-hmm", hmm)
    cfg.set_string("-dict", dictionary)
    cfg.set_string("-lm", lm)
    cfg.set_string("-logfn", "/dev/null")
    cfg.set_string("-wip", "0.65")     # word insertion penalty (default)
    cfg.set_string("-wdp", "0.80")     # word deletion penalty (default)
    if extra:
        for k, v in extra.items():
            cfg.set_string(k, v)
    dec = Decoder(cfg)
    if jsgf:
        with open(jsgf) as f:
            dec.add_jsgf_string("vcm", f.read())
        dec.activate_search("vcm")
    return dec


def decode_clip(dec, path):
    """Return (transcript, latency_ms, utt_prob)."""
    t0 = time.perf_counter()
    dec.start_utt()
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getsampwidth() == 2, f"expected 16-bit, got {w.getsampwidth()}"
        while True:
            frames = w.readframes(4000)
            if not frames:
                break
            dec.process_raw(frames, False, False)
    dec.end_utt()
    dt = (time.perf_counter() - t0) * 1000.0
    hyp = dec.hyp()
    text = (hyp.hypstr or "").strip() if hyp else ""
    prob = dec.get_prob()
    return text, dt, prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hmm", required=True)
    ap.add_argument("--dict", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--jsgf", default=None)
    ap.add_argument("--data", default=os.path.join(_REPO, "data",
                                                   "additional_test_data"))
    ap.add_argument("--report", default=None)
    ap.add_argument("--extra", default=None,
                    help="comma-separated -key value pairs, e.g. '-silprob 0.5'")
    args = ap.parse_args()

    extra = {}
    if args.extra:
        toks = args.extra.split()
        for i in range(0, len(toks), 2):
            extra[toks[i]] = toks[i + 1]

    print(f"[{args.name}] building decoder (hmm={os.path.basename(os.path.dirname(args.hmm))} "
          f"jsgf={bool(args.jsgf)})", flush=True)
    dec = make_decoder(args.hmm, args.dict, args.lm, args.jsgf, extra)
    clf = load_classifier()

    rows = build_ground_truth(args.data)
    print(f"[{args.name}] {len(rows)} clips", flush=True)
    t0 = time.perf_counter()
    results = []
    for k, r in enumerate(rows):
        text, ms, prob = decode_clip(dec, r["path"])
        command, cprob = predict(clf, text)
        w = wer_breakdown(r["spoken"], text)
        results.append({
            "path": r["path"], "folder": r["folder"], "spoken": r["spoken"],
            "gold": r["gold"],
            "gold_intent": _INTENT_OF.get(r["gold"], "unknown"),
            "transcript": text, "pred": command,
            "pred_intent": _INTENT_OF.get(command, "unknown"),
            "prob": round(prob, 6), "clf_prob": round(cprob, 4),
            "wer": round(w, 3), "asr_ms": round(ms, 1),
        })
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  {k + 1}/{len(rows)}  "
                  f"({(time.perf_counter() - t0) / (k + 1):.2f}s/clip)  "
                  f"{text!r} -> {command}", flush=True)
    wall = time.perf_counter() - t0

    n = len(results)
    cmd_correct = sum(1 for x in results if x["pred"] == x["gold"])
    intent_correct = sum(1 for x in results if x["pred_intent"] == x["gold_intent"])
    blank = sum(1 for x in results if not x["transcript"].strip())
    reject_pred = sum(1 for x in results if x["pred"] == "REJECT")
    wers = sorted(x["wer"] for x in results)
    lats = sorted(x["asr_ms"] for x in results)
    per_folder = {}
    for f in sorted(set(x["folder"] for x in results)):
        fr = [x for x in results if x["folder"] == f]
        per_folder[f] = {
            "n": len(fr),
            "cmd_acc": round(sum(1 for x in fr if x["pred"] == x["gold"]) / len(fr), 4),
            "intent_acc": round(sum(1 for x in fr if x["pred_intent"] == x["gold_intent"]) / len(fr), 4),
        }
    summary = {
        "model": f"PocketSphinx [{args.name}]",
        "hmm": args.hmm, "dict": args.dict, "lm": args.lm,
        "jsgf": args.jsgf, "extra": extra,
        "n_clips": n,
        "command_acc": round(cmd_correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "blank_rate": round(blank / n, 4),
        "reject_pred": reject_pred,
        "mean_wer": round(float(np.mean(wers)), 4),
        "median_wer": round(float(np.median(wers)), 4),
        "p90_wer": round(wers[int(0.9 * (n - 1))], 4),
        "asr_ms_p50": round(lats[n // 2], 1),
        "asr_ms_p95": round(lats[int(0.95 * (n - 1))], 1),
        "wall_s": round(wall, 1),
        "per_folder": per_folder,
        "results": results,
    }
    print(f"\n[{args.name}] command_acc={summary['command_acc']:.4f} "
          f"intent_acc={summary['intent_acc']:.4f} "
          f"mean_wer={summary['mean_wer']:.3f} median_wer={summary['median_wer']:.3f} "
          f"blank={summary['blank_rate']:.3f} p50={summary['asr_ms_p50']}ms", flush=True)
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{args.name}] report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
