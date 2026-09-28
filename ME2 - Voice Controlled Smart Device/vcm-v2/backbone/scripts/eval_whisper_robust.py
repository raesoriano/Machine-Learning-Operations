#!/usr/bin/env python3
"""Robust zero-shot Whisper eval on additional_test_data.

Same metrics/ground-truth as eval_whisper.py, but each clip is transcribed by
a persistent worker process with a per-clip wall-clock timeout. A Whisper
repetition loop (the C++ ctranslate2 backend can spin forever on a noisy clip
even with condition_on_previous_text=False) is caught: the worker is killed
and respawned, and the clip is recorded as a blank + "timeout" flag instead of
hanging the whole run.

Usage:
    python backbone/scripts/eval_whisper_robust.py \
        --data ../test_data/additional_test_data \
        --report backbone/reports/additional_test_whisper.json \
        --size base.en --device cuda
"""
import argparse
import json
import os
import sys
import time
import multiprocessing as mp

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_ME2 = os.path.dirname(_REPO)  # vcm-v2 lives inside the ME2 folder
_SANDBOX = os.path.dirname(os.path.dirname(_ME2))  # .../sandbox
sys.path.insert(0, _ME2)          # for `vcm` (normalize -> vcm.parser)
sys.path.insert(0, _BACK)
sys.path.insert(0, os.path.join(_REPO, "archive", "ctc_v8"))
sys.path.insert(0, _HERE)

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
    "CALL": "call", "MESSAGE": "call", "REJECT": "unknown",
}


def wer_breakdown(ref, hyp):
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


def _worker(task_q, result_q, size, device, compute):
    """Persistent worker: load whisper once, transcribe clips until told to quit."""
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device=device, compute_type=compute)

    def transcribe(path):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            import torchaudio
            a = torchaudio.functional.resample(
                __import__("torch").from_numpy(a), sr, 16000).numpy().astype(np.float32)
        t0 = time.perf_counter()
        segs, _ = model.transcribe(
            a, language="en", beam_size=1, vad_filter=False,
            condition_on_previous_text=False, no_speech_threshold=0.6)
        text = " ".join(s.text.strip() for s in segs)
        return normalize(text), (time.perf_counter() - t0) * 1e3

    while True:
        item = task_q.get()
        if item is None:
            break
        idx, path = item
        try:
            text, ms = transcribe(path)
            result_q.put((idx, text, ms, False))
        except Exception as e:  # noqa: BLE001
            result_q.put((idx, "", 0.0, True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--size", default="base.en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute", default="float16")
    ap.add_argument("--clip-timeout", type=float, default=25.0)
    args = ap.parse_args()

    clf = load_classifier(os.path.join(_REPO, "archive", "ctc_v8",
                                       "artifacts", "classifier.pkl"))
    rows = build_ground_truth(args.data)
    print(f"whisper {args.size} robust eval on {len(rows)} clips "
          f"(clip timeout {args.clip_timeout}s) ...", flush=True)

    ctx = mp.get_context("spawn")
    trans = {}
    asr_ms = {}
    timeouts = []
    worker = None
    task_q = result_q = None

    def start_worker():
        nonlocal worker, task_q, result_q
        task_q = ctx.Queue()
        result_q = ctx.Queue()
        worker = ctx.Process(target=_worker,
                             args=(task_q, result_q, args.size, args.device,
                                   args.compute), daemon=True)
        worker.start()

    start_worker()
    t0 = time.perf_counter()
    for k, r in enumerate(rows):
        task_q.put((k, r["path"]))
        try:
            idx, text, ms, _err = result_q.get(timeout=args.clip_timeout)
            trans[k] = text
            asr_ms[k] = ms
        except Exception:  # queue.Empty -> clip hung
            timeouts.append(r["path"])
            trans[k] = ""
            asr_ms[k] = 0.0
            print(f"  [timeout {args.clip_timeout:.0f}s] clip {k + 1} "
                  f"{os.path.basename(r['path'])} -> respawn worker", flush=True)
            worker.terminate()
            worker.join()
            start_worker()
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  {k + 1}/{len(rows)}  "
                  f"({(time.perf_counter() - t0) / (k + 1):.2f}s/clip)  "
                  f"{trans[k]!r}", flush=True)
    task_q.put(None)
    worker.join(timeout=10)
    if worker.is_alive():
        worker.terminate()
    wall = time.perf_counter() - t0

    # ---- metrics (same convention as eval_whisper.py) ----
    results = []
    for k, r in enumerate(rows):
        text = trans[k]
        command, prob = predict(clf, text)
        w = wer_breakdown(r["spoken"], text)
        results.append({
            "path": r["path"], "folder": r["folder"], "spoken": r["spoken"],
            "gold": r["gold"],
            "gold_intent": _INTENT_OF.get(r["gold"], "unknown"),
            "transcript": text, "pred": command,
            "pred_intent": _INTENT_OF.get(command, "unknown"),
            "prob": round(prob, 4), "wer": round(w, 3),
            "asr_ms": round(asr_ms[k], 1),
            "timeout": k in [timeouts.index(t) for t in timeouts] if timeouts else False,
        })
    n = len(results)
    cmd_correct = sum(1 for x in results if x["pred"] == x["gold"])
    intent_correct = sum(1 for x in results if x["pred_intent"] == x["gold_intent"])
    blank = sum(1 for x in results if not x["transcript"].strip())
    reject_pred = sum(1 for x in results if x["pred"] == "REJECT")
    wer_sum = sum(x["wer"] for x in results)
    asr_lat = sorted(x["asr_ms"] for x in results if x["asr_ms"] > 0)
    report = {
        "model": f"whisper {args.size} (faster-whisper, zero-shot)",
        "n_clips": n, "command_acc": round(cmd_correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "blank_rate": round(blank / n, 4),
        "reject_pred": reject_pred,
        "mean_wer": round(wer_sum / n, 4),
        "timeouts": len(timeouts),
        "timeout_clips": [os.path.basename(t) for t in timeouts],
        "asr_ms_p50": round(asr_lat[len(asr_lat) // 2], 1) if asr_lat else None,
        "asr_ms_p95": round(asr_lat[int(0.95 * len(asr_lat))], 1) if asr_lat else None,
        "wall_s": round(wall, 1),
        "per_folder": {},
        "clips": results,
    }
    for f in sorted(set(x["folder"] for x in results)):
        sub = [x for x in results if x["folder"] == f]
        report["per_folder"][f] = {
            "n": len(sub),
            "cmd_acc": round(sum(1 for x in sub if x["pred"] == x["gold"]) / len(sub), 4),
            "intent_acc": round(sum(1 for x in sub if x["pred_intent"] == x["gold_intent"]) / len(sub), 4),
        }
    print(json.dumps({k: report[k] for k in
                      ("command_acc", "intent_acc", "blank_rate", "mean_wer",
                       "timeouts", "asr_ms_p50", "asr_ms_p95")}, indent=2))
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
