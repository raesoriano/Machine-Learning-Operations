#!/usr/bin/env python3
"""pi test v6 -- evaluate the HMM/GMM recognizer on the 176-clip held-out test
set with the SAME ground truth and metric definitions as v3
(training/eval_pocketsphinx.py), so numbers are directly comparable.

Pipeline per clip:
    waveform -> features -> constrained decode (grammar FSA x phone HMMs)
             -> best phrase -> command class (commands.phrase_to_class)
    plus a free decode (unigram) -> "what I actually heard"
    REJECT if the grammar decode is implausible (low score) or the free decode
    is far better (the utterance is not one of our commands).

Report keys match v3: model, n_clips, command_acc, intent_acc, blank_rate,
mean_wer, asr_ms_p50, asr_ms_p90, per_folder, results.
"""
from __future__ import annotations
import os, sys, json, time, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hgm.feats import read_wav, features
from hgm.acoustic import AcousticModel
from hgm.dict import load as load_dict
from hgm.lm import LanguageModel
from hgm.decode import Decoder
from hgm.commands import phrase_to_class, class_to_intent

ME2 = os.path.dirname(HERE)
TEST = os.path.join(ME2, "data", "additional_test_data")
MODELS = os.path.join(HERE, "models")

# test-folder -> gold command class (32-class ontology)
FOLDER_CLASS = {
    "ALARM": "ALARM", "BRIGHTNESS": "BRIGHTNESS", "CALL": "CALL",
    "COLOR": "COLOR", "CREATE_REMINDER": "CREATE_REMINDER",
    "LIGHT_OFF": "LIGHT_OFF", "LIGHT_ON": "LIGHT_ON",
    "LIST_REMINDERS": "LIST_REMINDERS", "MESSAGE": "MESSAGE", "NEXT": "NEXT",
    "PAUSE": "PAUSE", "PLAY_MUSIC": "PLAY_MUSIC", "REJECT": "REJECT",
    "STOP": "STOP", "TEMPERATURE": "TEMPERATURE", "TIME": "TIME",
    "TIMER": "TIMER", "VOLUME_DOWN": "VOLUME_DOWN", "VOLUME_UP": "VOLUME_UP",
    "WEATHER": "WEATHER",
}
# gold class (coarse) -> set of fine classes that count as correct
COARSE_OK = {
    "ALARM": {"ALARM", "ALARM_6_00AM", "ALARM_8_00AM", "ALARM_9_00PM"},
    "BRIGHTNESS": {"BRIGHTNESS", "BRIGHTNESS_100", "BRIGHTNESS_60", "BRIGHTNESS_20"},
    "COLOR": {"COLOR", "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE"},
    "TEMPERATURE": {"TEMPERATURE", "TEMPERATURE_18", "TEMPERATURE_22", "TEMPERATURE_26"},
    "CREATE_REMINDER": {"CREATE_REMINDER", "CREATE_REMINDER_DRINK_WATER",
                        "CREATE_REMINDER_EXERCISE", "CREATE_REMINDER_STUDY"},
    "TIMER": {"TIMER", "TIMER_1m", "TIMER_30s", "TIMER_10s"},
}
for k, v in list(COARSE_OK.items()):
    COARSE_OK[k] = v | {k}


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    import numpy as _np
    n, m = len(r), len(h)
    dp = _np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m] / max(1, n)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--beam", type=int, default=4000)
    ap.add_argument("--reject-margin", type=float, default=8.0,
                    help="reject if free_lp - grammar_lp > margin (per-frame)")
    ap.add_argument("--report", default=os.path.join(HERE, "test_v6_report.json"))
    args = ap.parse_args()

    am = AcousticModel.load(os.path.join(MODELS, "acoustic_model.npz"))
    dictionary = load_dict(os.path.join(MODELS, "dictionary.txt"))
    lm = LanguageModel.load(os.path.join(MODELS, "lm.npz"))
    dec = Decoder(am, dictionary, lm.words, beam=args.beam)
    print(f"loaded: {am.n_phones} phones, {len(dictionary)} words, "
          f"{lm.n_phrases} phrases", flush=True)

    rows = []
    for folder in sorted(os.listdir(TEST)):
        d = os.path.join(TEST, folder)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.wav"))):
            rows.append((folder, p))
    print(f"test clips: {len(rows)}", flush=True)

    results = []
    t0 = time.time()
    for k, (folder, path) in enumerate(rows):
        gold = FOLDER_CLASS.get(folder, "REJECT")
        x = read_wav(path)
        F = features(x)
        T = F.shape[0]
        t1 = time.perf_counter()
        words, glp, pidx = dec.decode_constrained(lm, F)
        fwords, flp = dec.decode_free(lm, F)
        asr_ms = (time.perf_counter() - t1) * 1000
        # normalize scores per frame for a length-independent margin
        glp_pf = glp / max(1, T)
        flp_pf = flp / max(1, T)
        phrase = " ".join(words)
        pred = phrase_to_class(phrase) if words else "REJECT"
        # REJECT decision: no phrase, or free decode clearly better
        if not words or (flp_pf - glp_pf) > args.reject_margin:
            pred = "REJECT"
        correct = (pred == gold) or (gold in COARSE_OK and pred in COARSE_OK[gold])
        results.append({
            "path": os.path.relpath(path, TEST), "folder": folder,
            "gold": gold, "heard": " ".join(fwords),
            "transcript": phrase, "pred": pred,
            "pred_intent": class_to_intent(pred),
            "gold_intent": class_to_intent(gold),
            "correct": bool(correct),
            "wer": _wer(phrase, " ".join(fwords)) if words else 1.0,
            "asr_ms": asr_ms,
        })
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    n = len(results)
    cmd_correct = sum(r["correct"] for r in results)
    intent_correct = sum(r["pred_intent"] == r["gold_intent"] for r in results)
    blank = sum(1 for r in results if not r["transcript"])
    ms = sorted(r["asr_ms"] for r in results)
    per_folder = {}
    for folder in sorted(set(r["folder"] for r in results)):
        sub = [r for r in results if r["folder"] == folder]
        per_folder[folder] = {
            "n": len(sub),
            "cmd_acc": round(sum(r["correct"] for r in sub) / len(sub), 4),
            "intent_acc": round(sum(r["pred_intent"] == r["gold_intent"] for r in sub) / len(sub), 4),
        }
    report = {
        "model": "hmm-gmm-v6",
        "n_clips": n,
        "command_acc": round(cmd_correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "blank_rate": round(blank / n, 4),
        "mean_wer": round(sum(r["wer"] for r in results) / n, 4),
        "asr_ms_p50": round(ms[n // 2], 1),
        "asr_ms_p90": round(ms[int(n * 0.9)], 1),
        "reject_margin": args.reject_margin,
        "per_folder": per_folder,
        "results": results,
    }
    json.dump(report, open(args.report, "w"), indent=1)
    print(f"\n[{report['model']}] command_acc={report['command_acc']:.4f} "
          f"intent_acc={report['intent_acc']:.4f} "
          f"blank={report['blank_rate']:.3f} wer={report['mean_wer']:.3f} "
          f"asr_p50={report['asr_ms_p50']:.0f}ms p90={report['asr_ms_p90']:.0f}ms", flush=True)
    print(f"report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
