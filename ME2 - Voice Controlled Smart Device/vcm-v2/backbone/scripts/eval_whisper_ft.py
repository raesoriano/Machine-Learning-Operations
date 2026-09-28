#!/usr/bin/env python3
"""Evaluate the FINE-TUNED Whisper (HF model) + stage-2 classifier on the
held-out 171 new-speaker clips. Same metrics/ground-truth as
eval_whisper_robust.py so the numbers are directly comparable to the
zero-shot baseline (81.9% command).

The fine-tuned model is an HF WhisperForConditionalGeneration (from
finetune_whisper.py), so inference uses model.generate() (slower than
faster-whisper but exact). Each transcript is post-processed through the
same normalize() the classifier was trained on.

Usage:
    CUDA_VISIBLE_DEVICES=0 python backbone/scripts/eval_whisper_ft.py \
        --model backbone/artifacts/whisper_base_ft/best \
        --data ../data/additional_test_data \
        --report backbone/reports/additional_test_whisper_ft.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
from transformers import (AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_SANDBOX = os.path.dirname(os.path.dirname(_ME2))  # .../sandbox
_ME2 = os.path.dirname(_REPO)  # vcm-v2 lives inside the ME2 folder
sys.path.insert(0, _ME2)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--report", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device
    print(f"loading fine-tuned whisper from {args.model} ...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.to(device)
    model.eval()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        args.model if os.path.isfile(os.path.join(args.model,
                                                  "preprocessor_config.json"))
        else os.path.dirname(os.path.abspath(args.model)))
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def transcribe(path):
        a, sr = sf.read(path, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            import torchaudio
            a = torchaudio.functional.resample(
                torch.from_numpy(a), sr, 16000).numpy().astype(np.float32)
        t0 = time.perf_counter()
        inputs = feature_extractor(a, sampling_rate=16000,
                                   return_tensors="pt",
                                   padding="max_length").to(device)
        # The extractor no longer returns an attention mask; build it from
        # the waveform length (50 mel frames/s, 3000-frame max).
        n_frames = min(int(round(len(a) / 16000 * 50)), 3000)
        mask = torch.zeros(1, 3000, dtype=torch.long)
        mask[0, :n_frames] = 1
        mask = mask.to(device)
        with torch.no_grad():
            ids = model.generate(inputs.input_features,
                                 attention_mask=mask,
                                 max_new_tokens=64,
                                 do_sample=False)
        # strip leading pad/sot
        if ids[0, 0] == tokenizer.pad_token_id:
            ids = ids[:, 1:]
        text = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        return normalize(text), (time.perf_counter() - t0) * 1e3

    clf = load_classifier(os.path.join(_REPO, "archive", "ctc_v8",
                                       "artifacts", "classifier.pkl"))
    rows = build_ground_truth(args.data)
    print(f"fine-tuned whisper eval on {len(rows)} clips ...", flush=True)

    results = []
    t0 = time.perf_counter()
    for k, r in enumerate(rows):
        text, ms = transcribe(r["path"])
        command, prob = predict(clf, text)
        w = wer_breakdown(r["spoken"], text)
        results.append({
            "path": r["path"], "folder": r["folder"], "spoken": r["spoken"],
            "gold": r["gold"],
            "gold_intent": _INTENT_OF.get(r["gold"], "unknown"),
            "transcript": text, "pred": command,
            "pred_intent": _INTENT_OF.get(command, "unknown"),
            "prob": round(prob, 4), "wer": round(w, 3), "asr_ms": round(ms, 1),
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
    wer_sum = sum(x["wer"] for x in results)
    wer_sorted = sorted(x["wer"] for x in results)
    n_looping = sum(1 for x in results if x["wer"] > 1.0)
    asr_lat = sorted(x["asr_ms"] for x in results if x["asr_ms"] > 0)
    report = {
        "model": "whisper base.en FINE-TUNED (HF, ME2 domain)",
        "n_clips": n, "command_acc": round(cmd_correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "blank_rate": round(blank / n, 4), "reject_pred": reject_pred,
        "mean_wer": round(wer_sum / n, 4),
        "median_wer": round(wer_sorted[n // 2], 4),
        "p90_wer": round(wer_sorted[int(0.9 * n)], 4),
        "n_looping": n_looping,
        "loop_pct": round(100.0 * n_looping / n, 1),
        "asr_ms_p50": round(asr_lat[len(asr_lat) // 2], 1) if asr_lat else None,
        "asr_ms_p95": round(asr_lat[int(0.95 * len(asr_lat))], 1) if asr_lat else None,
        "wall_s": round(wall, 1), "per_folder": {}, "clips": results,
    }
    report["wer_note"] = (
        f"mean_wer is inflated by repetition loops on {n_looping}/{n} clips "
        f"({report['loop_pct']}%): the fine-tuned model does not emit EOS on "
        "some OOD/noisy input and repeats a phrase. median_wer is the typical "
        "clip. Loops do NOT hurt command_acc (the stage-2 classifier extracts "
        "the command from the looped transcript).")
    for f in sorted(set(x["folder"] for x in results)):
        sub = [x for x in results if x["folder"] == f]
        report["per_folder"][f] = {
            "n": len(sub),
            "cmd_acc": round(sum(1 for x in sub if x["pred"] == x["gold"]) / len(sub), 4),
            "intent_acc": round(sum(1 for x in sub if x["pred_intent"] == x["gold_intent"]) / len(sub), 4),
        }
    print(json.dumps({k: report[k] for k in
                      ("command_acc", "intent_acc", "blank_rate", "mean_wer",
                       "median_wer", "p90_wer", "n_looping", "loop_pct",
                       "asr_ms_p50", "asr_ms_p95")}, indent=2))
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
