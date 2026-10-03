#!/usr/bin/env python3
"""Download the ME2 dataset from Hugging Face into the layout the v8 code reads.

Source: https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands

Layout written (under --out, default ./dataset):

    dataset/
      train/    manifest.csv  audio/*.wav
      test/     manifest.csv  audio/*.wav
      holdout/  manifest.csv  audio/*.wav
      numerals/ manifest.csv  audio/*.wav
      synthetic_negatives/    manifest.csv  audio/*.wav   (with --negatives)

Each manifest.csv keeps the HF columns (file, transcript, command, variation,
slot_value, out_of_scope, speaker_id, source, is_synthetic, ...); `file` is the
basename the v8 code joins against <split>/audio/. Audio is written as 16 kHz
16-bit mono WAV (the format data.py reads).

Usage:
    python download_dataset.py                 # train+test+holdout+numerals (~7 GB)
    python download_dataset.py --negatives     # also the 1,250 synthetic negatives
    python download_dataset.py --splits train test --out /path/to/dataset

Needs: pip install datasets pandas soundfile
"""
from __future__ import annotations
import argparse
import os
import sys
import time

import pandas as pd

HF_ID = "airimonda/ai231-me2-voice-commands"
DEFAULT_SPLITS = ["train", "test", "holdout", "numerals"]
NEG_CONFIG = "synthetic_negatives"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS,
                    help="HF splits to download (default: %s)" % " ".join(DEFAULT_SPLITS))
    ap.add_argument("--negatives", action="store_true",
                    help="also download the synthetic_negatives config (train+test)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "dataset"))
    args = ap.parse_args()

    from datasets import load_dataset
    import soundfile as sf

    jobs = [(s, "default") for s in args.splits]
    if args.negatives:
        jobs += [("train", NEG_CONFIG), ("test", NEG_CONFIG)]

    for split, config in jobs:
        subdir = NEG_CONFIG if config != "default" else split
        dest = os.path.join(args.out, subdir)
        audio_dir = os.path.join(dest, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        print(f"=== {config}/{split} -> {dest} ===", flush=True)
        t0 = time.time()
        ds = load_dataset(HF_ID, config, split=split)
        rows = []
        for i, r in enumerate(ds):
            arr, sr = r["audio"]["array"], r["audio"]["sampling_rate"]
            base = os.path.basename(r["file"])
            sf.write(os.path.join(audio_dir, base), arr, sr, subtype="PCM_16")
            row = {k: v for k, v in r.items() if k != "audio"}
            row["file"] = base
            rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(ds)}  ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(dest, "manifest.csv"), index=False)
        print(f"  done: {len(rows)} clips in {time.time() - t0:.0f}s", flush=True)

    print(f"\nAll done. Point the v8 code at it with:  --data {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
