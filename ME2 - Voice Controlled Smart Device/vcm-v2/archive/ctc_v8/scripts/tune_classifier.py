#!/usr/bin/env python3
"""A/B test classifier configs on the HELD-OUT `val` split (not the 171-clip set).

Methodology fix: the previous train_classifier.py trained on ALL 82,778 rows
(train+val+test combined) and reported in-sample accuracy (99.87%) -- meaningless
for tuning. This script:

  * trains on the manifest `train` split only
  * evaluates each candidate config on the held-out `val` split (5,318 pos +
    2,939 neg)  <-- all tuning decisions are made here
  * reports command accuracy + REJECT behaviour on val

Candidate axes (all general, none fit to the 171-clip test set):
  * word ngram range (1,2) vs (1,3)
  * + char_wb ngrams (2,5)   -> tolerance to typos / ASR errors (e.g. "voluyme")
  * class_weight balanced vs None  -> fixes LIGHT_ON vs LIGHT_OFF imbalance and
                                      the dominant STOP ("stop" x4167)
  * C (inverse regularisation) in {1, 4, 16}

The winning config (by val command accuracy, tie-broken by lower false-REJECT
on positives) is then used to retrain the production model on ALL data.
"""
import csv
import itertools
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

from vcm2.normalize import normalize
from vcm2.classifier import REJECT

MAN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "manifests", "positive_negative_manifest.csv")


def load_split(split):
    texts, labels = [], []
    with open(MAN) as f:
        for r in csv.DictReader(f):
            if r["split"] != split:
                continue
            t = r["transcript"].strip()
            if not t:
                continue
            texts.append(t)
            labels.append(r["command"] if r["polarity"] == "positive" else REJECT)
    return texts, labels


def build(word_ngram=(1, 2), char_ngram=None, class_weight=None, C=4.0,
          min_df=3):
    steps = [("norm", FunctionTransformer(normalize, validate=False))]
    vecs = [("word", TfidfVectorizer(
        analyzer="word", ngram_range=word_ngram, sublinear_tf=True,
        min_df=min_df, strip_accents="unicode"))]
    if char_ngram:
        vecs.append(("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=char_ngram, sublinear_tf=True,
            min_df=min_df, strip_accents="unicode")))
    steps.append(("vec", FeatureUnion(vecs)))
    steps.append(("clf", LogisticRegression(C=C, max_iter=2000,
                                            solver="lbfgs",
                                            class_weight=class_weight)))
    return Pipeline(steps)


def eval_cfg(cfg, tr_x, tr_y, va_x, va_y):
    t0 = time.time()
    pipe = build(**cfg)
    pipe.fit(tr_x, tr_y)
    train_fit_s = time.time() - t0
    pred = pipe.predict(va_x)
    n = len(va_y)
    cmd_acc = sum(p == g for p, g in zip(pred, va_y)) / n
    # false-REJECT: positives (non-REJECT gold) that the model REJECTs
    pos_idx = [i for i, g in enumerate(va_y) if g != REJECT]
    fr = sum(pred[i] == REJECT for i in pos_idx) / len(pos_idx)
    # false-ACCEPT: REJECT gold that the model accepts as a command
    neg_idx = [i for i, g in enumerate(va_y) if g == REJECT]
    fa = sum(pred[i] != REJECT for i in neg_idx) / len(neg_idx) if neg_idx else 0.0
    return cmd_acc, fr, fa, train_fit_s


def main():
    print("loading splits ...", flush=True)
    tr_x, tr_y = load_split("train")
    va_x, va_y = load_split("val")
    print(f"  train: {len(tr_x)}  val: {len(va_x)}")

    grid = list(itertools.product(
        [(1, 2), (1, 3)],                       # word ngram
        [None, (2, 5)],                          # char ngram
        [None, "balanced"],                      # class weight
        [1.0, 4.0, 16.0],                        # C
    ))
    results = []
    for wn, cn, cw, C in grid:
        cfg = dict(word_ngram=wn, char_ngram=cn, class_weight=cw, C=C)
        acc, fr, fa, fit_s = eval_cfg(cfg, tr_x, tr_y, va_x, va_y)
        results.append({**cfg, "val_cmd_acc": round(acc, 4),
                        "val_false_reject": round(fr, 4),
                        "val_false_accept": round(fa, 4),
                        "fit_s": round(fit_s, 1)})
        print(f"  wn={str(wn):7s} char={str(cn):8s} cw={str(cw):8s} C={C:5.1f}  "
              f"val_acc={acc:6.4f}  false_rej={fr:5.4f}  false_acc={fa:5.4f}  "
              f"({fit_s:.1f}s)", flush=True)

    # rank: primary val_cmd_acc, secondary lower false_reject
    results.sort(key=lambda r: (-r["val_cmd_acc"], r["val_false_reject"]))
    print("\n=== ranked by val_cmd_acc (then lower false_reject) ===")
    for i, r in enumerate(results[:8]):
        print(f"  {i+1}. {r}")
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "reports", "tune_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
