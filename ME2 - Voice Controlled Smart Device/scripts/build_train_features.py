#!/usr/bin/env python3
"""Convert the ME2 positive/negative manifest into VCM training splits.

Outputs (in --out, default data/features):
  train.jsonl / val.jsonl / test.jsonl
      {id, audio (abs path), text (the TRANSCRIPT — what the audio actually
       says), intent, slots}
      positives only (CTC trains on in-domain commands; rejection is a
      decoder-side confidence threshold, not a CTC class)
  train.npz / val.npz / test.npz
      precomputed log-mel features {id: (T,40) float32 @16kHz} + token ids
      {id: [1-based word ids]} so training never re-reads audio.

Plan A: the CTC target is the transcript (the spoken words), NOT the
canonical target phrase. The old code trained on `target` (e.g.
"set an alarm for six am") while the audio said a paraphrase (e.g.
"Alarm 6 AM") — an unlearnable audio->text mapping that capped train
exact-match at ~17%. Training on the transcript makes the mapping
consistent; the parser (vcm.parser) then recovers (intent, slots).

Usage:
  python scripts/build_train_features.py \
      --manifest "/path/ME2.../data/manifests/positive_negative_manifest.csv" \
      --data-root "/path/ME2.../data" \
      --out data/features
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from vcm.features import log_mel  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data-root", required=True,
                    help="dir that manifest audio paths are relative to")
    ap.add_argument("--out", default=os.path.join(_REPO, "data", "features"))
    ap.add_argument("--max-rows", type=int, default=None,
                    help="debug: cap rows per split")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    splits = {"train": [], "val": [], "test": []}
    for r in rows:
        if r["polarity"] != "positive":
            continue
        audio = r["audio"]
        if not os.path.isabs(audio):
            audio = os.path.join(args.data_root, audio)
        splits[r["split"]].append({
            "id": r["id"],
            "audio": audio,
            "text": r["transcript"],   # Plan A: train on the spoken transcript
            "intent": r["intent"],
            "slots": json.loads(r["slots"]),
        })
    if args.max_rows:
        for k in splits:
            splits[k] = splits[k][: args.max_rows]

    from model.train import tokenize  # reuse the exact training tokenizer

    for name, rs in splits.items():
        t0 = time.time()
        jsonl = os.path.join(args.out, f"{name}.jsonl")
        with open(jsonl, "w") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        # precompute features (concatenated arrays + offsets; ragged-safe)
        ids, mel_chunks, tok_chunks = [], [], []
        n_bad = 0
        for i, r in enumerate(rs):
            try:
                from model.train import _load_audio
                m = log_mel(_load_audio(r["audio"]))
                t = tokenize(r["text"])
                if m.shape[0] < 4 or not t:
                    n_bad += 1
                    continue
                ids.append(r["id"])
                mel_chunks.append(m.astype(np.float32))
                tok_chunks.append(np.array(t, dtype=np.int32))
            except Exception as e:  # unreadable audio etc.
                n_bad += 1
                print(f"  skip {r['id']}: {e}")
        mel_off = np.zeros(len(ids) + 1, dtype=np.int64)
        tok_off = np.zeros(len(ids) + 1, dtype=np.int64)
        for i, (m, t) in enumerate(zip(mel_chunks, tok_chunks)):
            mel_off[i + 1] = mel_off[i] + m.shape[0]
            tok_off[i + 1] = tok_off[i] + t.shape[0]
        np.savez(os.path.join(args.out, f"{name}.npz"),
                 ids=np.array(ids),
                 mel_data=np.concatenate(mel_chunks, axis=0),
                 mel_off=mel_off,
                 tok_data=np.concatenate(tok_chunks, axis=0),
                 tok_off=tok_off)
        print(f"{name}: {len(rs)} rows -> {len(ids)} features, "
              f"{n_bad} skipped ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
