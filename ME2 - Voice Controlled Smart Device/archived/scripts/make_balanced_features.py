#!/usr/bin/env python3
"""Build an intent-balanced copy of the precomputed training features.

Per-intent balancing (README "next steps" lever 3): the training set is
heavily imbalanced (media_control ~11.7k rows vs call ~0.9k). We resample
training rows with replacement using inverse-intent-frequency weights (capped
so no intent is repeated >cap_up x or dropped below cap_down x), which lifts
the rare intents (call, set_timer, set_temperature, set_alarm) and trims the
dominant one (media_control) without throwing away most of the data.

The output is a drop-in replacement feature dir (same npz schema:
ids/mel_data/mel_off/tok_data/tok_off) so model.train runs unchanged:

    python scripts/make_balanced_features.py \
        --features data/features --out data/features_balanced

val/test are copied through untouched (balancing is a training-only lever).
"""
import argparse
import json
import os
import shutil
from collections import Counter

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="data/features_balanced")
    ap.add_argument("--cap-up", type=float, default=3.0,
                    help="max upsampling weight for a rare intent")
    ap.add_argument("--cap-down", type=float, default=0.5,
                    help="min downsampling weight for a common intent")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src = os.path.join(args.features, "train.npz")
    d = np.load(src)
    ids, mel_data, mel_off = d["ids"], d["mel_data"], d["mel_off"]
    tok_data, tok_off = d["tok_data"], d["tok_off"]
    n = len(ids)

    # intent per training row id
    intent = {}
    with open(os.path.join(args.features, "train.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                intent[r["id"]] = r.get("intent", "unknown")

    counts = Counter(intent.get(ids[i], "unknown") for i in range(n))
    mean = n / len(counts)
    w = np.zeros(n, dtype=np.float64)
    for i in range(n):
        it = intent.get(ids[i], "unknown")
        c = counts.get(it, 1)
        w[i] = min(max(mean / c, args.cap_down), args.cap_up)
    p = w / w.sum()

    # resample n rows with replacement (same epoch size as the raw set)
    idx = rng.choice(n, size=n, replace=True, p=p)

    # rebuild the packed arrays in the new (balanced) order
    mel_slices = [mel_data[mel_off[idx[i]]:mel_off[idx[i] + 1]] for i in range(n)]
    new_mel_data = np.vstack(mel_slices)
    new_mel_off = np.concatenate(
        [[0], np.cumsum([m.shape[0] for m in mel_slices])]).astype(np.int64)
    tok_slices = [tok_data[tok_off[idx[i]]:tok_off[idx[i] + 1]] for i in range(n)]
    new_tok_data = np.concatenate(tok_slices)
    new_tok_off = np.concatenate(
        [[0], np.cumsum([t.shape[0] for t in tok_slices])]).astype(np.int64)
    new_ids = ids[idx]

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "train.npz"),
             ids=new_ids, mel_data=new_mel_data, mel_off=new_mel_off,
             tok_data=new_tok_data, tok_off=new_tok_off)
    for name in ("val", "test"):
        for ext in (".npz", ".jsonl"):
            s = os.path.join(args.features, name + ext)
            if os.path.exists(s):
                shutil.copy(s, os.path.join(args.out, name + ext))

    # report the effective per-intent counts in the balanced set
    bcounts = Counter(intent.get(new_ids[i], "unknown") for i in range(n))
    print(f"balanced train rows: {n}  (seed={args.seed}, "
          f"cap_up={args.cap_up}, cap_down={args.cap_down})")
    print(f"intent {chr(32)*10} raw {chr(32)*4} balanced")
    for k in sorted(counts, key=lambda k: -counts[k]):
        print(f"{k:16s} {counts[k]:7d} {bcounts.get(k, 0):9d}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
