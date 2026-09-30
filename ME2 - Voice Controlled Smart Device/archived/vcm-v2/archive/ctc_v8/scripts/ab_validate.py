#!/usr/bin/env python3
"""Controlled A/B on the HELD-OUT val split: isolate each change's effect.

  A  old config        (unbalanced, word 1-2 only)
  B  +balanced +char   (new config, same data)
  C  B + data-clean    (new config + lights-out relabel + STOP 'end X')

All trained on the `train` split, evaluated on `val`. This is the honest
per-change breakdown (the 171-clip set is held out for the final report).
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcm2.classifier import build_pipeline, load_rows, REJECT   # noqa: E402
from vcm2.data_clean import clean_rows                          # noqa: E402

MAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "manifests", "positive_negative_manifest.csv")


def main():
    tr_x, tr_y, _, _ = load_rows(MAN, "train")
    va_x, va_y, _, _ = load_rows(MAN, "val")

    with open(MAN) as f:
        allrows = list(csv.DictReader(f))
    cleaned, rep = clean_rows(allrows)

    def load_clean(split):
        t, l = [], []
        for r in cleaned:
            if r["split"] != split or not r["transcript"].strip():
                continue
            t.append(r["transcript"])
            l.append(r["command"] if r["polarity"] == "positive" else REJECT)
        return t, l

    ctr_x, ctr_y = load_clean("train")

    def ev(cfg, clean=False):
        x, y = (ctr_x, ctr_y) if clean else (tr_x, tr_y)
        p = build_pipeline(cfg)
        p.fit(x, y)
        pred = p.predict(va_x)
        n = len(va_y)
        acc = sum(a == b for a, b in zip(pred, va_y)) / n
        pos = [i for i, g in enumerate(va_y) if g != REJECT]
        fr = sum(pred[i] == REJECT for i in pos) / len(pos)
        neg = [i for i, g in enumerate(va_y) if g == REJECT]
        fa = sum(pred[i] != REJECT for i in neg) / len(neg)
        return acc, fr, fa

    OLD = dict(word_ngram=(1, 2), char_ngram=None, class_weight=None,
               C=4.0, min_df=3)
    NEW = dict(word_ngram=(1, 2), char_ngram=(2, 5), class_weight="balanced",
               C=4.0, min_df=3)
    print(f"{'config':38s} val_acc  false_rej  false_acc")
    for name, cfg, clean in [
        ("A old (unbalanced, word only)", OLD, False),
        ("B +balanced +char(2,5)", NEW, False),
        ("C B + data-clean", NEW, True)]:
        a, fr, fa = ev(cfg, clean)
        print(f"{name:38s} {a:7.4f}  {fr:9.4f}  {fa:9.4f}")
    print("\nclean report:", rep)


if __name__ == "__main__":
    main()
