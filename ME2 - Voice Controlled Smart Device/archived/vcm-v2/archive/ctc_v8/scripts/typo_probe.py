#!/usr/bin/env python3
"""Does the char-ngram width help on near-miss typos WITHOUT hurting val?

Trains on the `train` split, reports held-out `val` accuracy + false-REJECT,
and probes a set of plausible "volume up" typos (general, not just the one
test example). If a wider char ngram recovers typos with no val regression,
it's a principled robustness gain; if not, we keep (2,5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcm2.classifier import build_pipeline, load_rows, REJECT, predict  # noqa: E402

MAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "manifests", "positive_negative_manifest.csv")

PROBES = ["voluyme up", "volueme up", "volue up", "volyme up",
          "volum up", "volueem up", "turn the voluyme up"]


def main():
    tr_x, tr_y, _, _ = load_rows(MAN, "train")
    va_x, va_y, _, _ = load_rows(MAN, "val")
    base = dict(word_ngram=(1, 2), class_weight="balanced", C=4.0, min_df=3)
    print(f"{'char_ngram':12s} val_acc  false_rej   typo probes")
    for cn in [None, (2, 5), (2, 6), (1, 6), (2, 7)]:
        cfg = dict(base, char_ngram=cn)
        p = build_pipeline(cfg)
        p.fit(tr_x, tr_y)
        pred = p.predict(va_x)
        n = len(va_y)
        acc = sum(a == b for a, b in zip(pred, va_y)) / n
        pos = [i for i, g in enumerate(va_y) if g != REJECT]
        fr = sum(pred[i] == REJECT for i in pos) / len(pos)
        res = "  ".join(f"{q}={predict(p, q)[0][:6]}" for q in PROBES)
        print(f"{str(cn):12s} {acc:7.4f}  {fr:8.4f}  {res}")


if __name__ == "__main__":
    main()
