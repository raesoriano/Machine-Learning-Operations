#!/usr/bin/env python3
"""ASR-only regression check on the ME2 held-out test split.

Measures whether fine-tuning on the new-speaker subset hurt the ASR on the
original (already-seen) speakers. Reports word error rate (WER) and word
exact-match on the ME2 test split (6,470 clips) for a given checkpoint, so we
can compare me2_v6 (baseline) vs me2_v8 (fine-tuned) on the SAME held-out set.

This is the "did we regress the old data" guardrail. The new-speaker general-
ization gain is measured separately by scripts/eval.py on additional_test_data.

Usage:
    python scripts/eval_asr_split.py --ckpt artifacts/asr/me2_v8/best.pt \
        --split test --report reports/asr_regression.json
"""
import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))      # .../VCM-v2/scripts
_VCM2 = os.path.dirname(_HERE)                          # .../VCM-v2
_SANDBOX = os.path.dirname(_VCM2)                       # .../sandbox
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
sys.path.insert(0, _VCM2)
sys.path.insert(0, _ME2)

from vcm2.asr import ASR                       # noqa: E402
from model.train import load_features, _row_mel, _row_toks  # noqa: E402
from model.decode import greedy_decode         # noqa: E402
from model.model_def import BLANK              # noqa: E402


def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1))
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return d[len(r), len(h)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--features",
                    default=os.path.join(_ME2, "data", "features"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    asr = ASR(ckpt=args.ckpt, device="cpu")
    rows = load_features(args.features, args.split)
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"evaluating {len(rows)} {args.split} rows with "
          f"{os.path.basename(os.path.dirname(args.ckpt))}")

    tot_sub = tot_del = tot_ins = tot_ref = 0
    exact = 0
    for row in rows:
        mels = _row_mel(row)
        import torch
        with torch.no_grad():
            logits = asr.model(torch.from_numpy(mels)[None])
        pred = greedy_decode(logits[0].cpu().numpy(), blank=BLANK)
        gold = _row_toks(row)
        from vcm.vocab import ID2WORD
        hyp = [ID2WORD[i - 1] for i in pred if i > 0]
        ref = [ID2WORD[i - 1] for i in gold if i > 0]
        # edit distance on words
        r, h = ref, hyp
        d = np.zeros((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            d[i, 0] = i
        for j in range(len(h) + 1):
            d[0, j] = j
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                              d[i - 1, j - 1] + cost)
        tot_ref += len(r)
        # decompose via backtrace
        i, j = len(r), len(h)
        s = de = ins = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if r[i-1]==h[j-1] else 1):
                if r[i-1] != h[j-1]:
                    s += 1
                i -= 1; j -= 1
            elif j > 0 and d[i, j] == d[i, j - 1] + 1:
                ins += 1; j -= 1
            else:
                de += 1; i -= 1
        tot_sub += s
        tot_del += de
        tot_ins += ins
        if hyp == ref:
            exact += 1

    n = len(rows)
    out = {
        "ckpt": os.path.basename(os.path.dirname(args.ckpt)),
        "split": args.split,
        "n_rows": n,
        "wer": round((tot_sub + tot_del + tot_ins) / max(tot_ref, 1), 4),
        "substitution": round(tot_sub / max(tot_ref, 1), 4),
        "deletion": round(tot_del / max(tot_ref, 1), 4),
        "insertion": round(tot_ins / max(tot_ref, 1), 4),
        "word_exact_match": round(exact / n, 4),
    }
    print(json.dumps(out, indent=2))
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
