#!/usr/bin/env python3
"""Ensemble PocketSphinx eval on the 171-clip held-out test set.

Two acoustic models decode the SAME command grammar (enh3, 103 phrases):

  * custom  : our 200-senone LDA AM trained on the ME2 dataset (1.6 MB)
  * stock   : the bundled en-us AM (6.4 MB), far more diverse training data

The two AMs have complementary errors (custom nails CALL, stock nails the
rest). Each clip is decoded by BOTH, each transcript is mapped to a command
by the SAME stage-2 classifier (TF-IDF + LogReg, 31 commands + REJECT), and
the final command is chosen by a test-set-AGNOSTIC fusion rule:

  * if both AMs classify to the SAME command  -> use it (agreement)
  * otherwise                                 -> use the one whose stage-2
    classifier is more confident (higher clf_prob)

No ground-truth information is used at inference. command_acc / intent_acc /
WER / latency are computed identically to eval_pocketsphinx.py and to the
Whisper fine-tune eval, so the numbers are directly comparable.

Footprint (under JSGF, LM unused at decode): custom AM 1.6 MB + stock AM
6.4 MB + dict 1.2 KB + JSGF 2.8 KB ~= 8.0 MB.

Usage
-----
    python backbone/pocketsphinx/eval_pocketsphinx_ensemble.py \
        --name ensemble \
        --custom-hmm <...>/pocketsphinx_trained_lda \
        --stock-hmm <...>/model/en-us/en-us \
        --dict <...>/dict3 \
        --jsgf <...>/vcm_commands_enh3.jsgf \
        --report backbone/reports/pocketsphinx_ensemble.json
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_ME2 = os.path.dirname(_REPO)
sys.path.insert(0, _ME2)
sys.path.insert(0, os.path.join(_REPO, "archive", "ctc_v8"))
sys.path.insert(0, _HERE)

from vcm2.classifier import load_classifier, predict      # noqa: E402
from vcm2.ground_truth import build_ground_truth          # noqa: E402

from eval_pocketsphinx import (                          # noqa: E402
    _INTENT_OF, make_decoder, decode_clip, wer_breakdown,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--custom-hmm", required=True)
    ap.add_argument("--custom-dict", default=None,
                    help="dict for the custom AM (default: --dict)")
    ap.add_argument("--custom-lm", required=True)
    ap.add_argument("--custom-extra", default="-silprob 0.65 -wip 0.65")
    ap.add_argument("--stock-hmm", required=True)
    ap.add_argument("--stock-dict", default=None,
                    help="dict for the stock AM (default: --dict)")
    ap.add_argument("--stock-lm", required=True)
    ap.add_argument("--stock-extra", default="-silprob 0.45 -wip 0.60")
    ap.add_argument("--dict", required=True)
    ap.add_argument("--jsgf", required=True)
    ap.add_argument("--data", default=os.path.join(_REPO, "test_data",
                                                   "additional_test_data"))
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    def _extra(s):
        toks = s.split()
        return {toks[i]: toks[i + 1] for i in range(0, len(toks), 2)}

    print(f"[{args.name}] building custom + stock decoders", flush=True)
    dec_c = make_decoder(args.custom_hmm, args.custom_dict or args.dict,
                         args.custom_lm, args.jsgf, _extra(args.custom_extra))
    dec_s = make_decoder(args.stock_hmm, args.stock_dict or args.dict,
                         args.stock_lm, args.jsgf, _extra(args.stock_extra))
    clf = load_classifier()

    rows = build_ground_truth(args.data)
    print(f"[{args.name}] {len(rows)} clips", flush=True)

    def classify(text):
        cmd, cprob = predict(clf, text)
        return cmd, cprob, _INTENT_OF.get(cmd, "unknown")

    t0 = time.perf_counter()
    results = []
    for k, r in enumerate(rows):
        tc, mc, pc = decode_clip(dec_c, r["path"])
        ts, ms, ps = decode_clip(dec_s, r["path"])
        cc, cpc, ic = classify(tc)
        cs, cps, is_ = classify(ts)
        # fusion: agreement, else higher stage-2 classifier confidence
        if cc == cs:
            cmd, intent, cprob, transcript, dec_ms, dec_prob = cc, ic, cpc, tc, mc, pc
        elif cpc >= cps:
            cmd, intent, cprob, transcript, dec_ms, dec_prob = cc, ic, cpc, tc, mc, pc
        else:
            cmd, intent, cprob, transcript, dec_ms, dec_prob = cs, is_, cps, ts, ms, ps
        w = wer_breakdown(r["spoken"], transcript)
        results.append({
            "path": r["path"], "folder": r["folder"], "spoken": r["spoken"],
            "gold": r["gold"], "gold_intent": _INTENT_OF.get(r["gold"], "unknown"),
            "transcript": transcript, "pred": cmd,
            "pred_intent": _INTENT_OF.get(cmd, "unknown"),
            "custom_transcript": tc, "custom_pred": cc, "custom_clf": round(cpc, 4),
            "stock_transcript": ts, "stock_pred": cs, "stock_clf": round(cps, 4),
            "prob": round(dec_prob, 6), "clf_prob": round(cprob, 4),
            "wer": round(w, 3), "asr_ms": round(mc + ms, 1),
        })
        if (k + 1) % 20 == 0 or k == 0:
            print(f"  {k + 1}/{len(rows)}  "
                  f"({(time.perf_counter() - t0) / (k + 1):.2f}s/clip)  "
                  f"{transcript!r} -> {cmd}", flush=True)
    wall = time.perf_counter() - t0

    n = len(results)
    cmd_correct = sum(1 for x in results if x["pred"] == x["gold"])
    intent_correct = sum(1 for x in results if x["pred_intent"] == x["gold_intent"])
    blank = sum(1 for x in results if not x["transcript"].strip())
    reject_pred = sum(1 for x in results if x["pred"] == "REJECT")
    agree = sum(1 for x in results if x["custom_pred"] == x["stock_pred"])
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
        "model": f"PocketSphinx ensemble [{args.name}] (custom + stock, agree+clf fusion)",
        "custom_hmm": args.custom_hmm, "stock_hmm": args.stock_hmm,
        "dict": args.dict, "jsgf": args.jsgf,
        "custom_extra": args.custom_extra, "stock_extra": args.stock_extra,
        "fusion": "agreement, else higher stage-2 clf confidence (test-set-agnostic)",
        "n_clips": n,
        "command_acc": round(cmd_correct / n, 4),
        "intent_acc": round(intent_correct / n, 4),
        "blank_rate": round(blank / n, 4),
        "reject_pred": reject_pred,
        "am_agreement": round(agree / n, 4),
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
          f"blank={summary['blank_rate']:.3f} agree={summary['am_agreement']:.3f} "
          f"p50={summary['asr_ms_p50']}ms", flush=True)
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[{args.name}] report -> {args.report}", flush=True)


if __name__ == "__main__":
    main()
