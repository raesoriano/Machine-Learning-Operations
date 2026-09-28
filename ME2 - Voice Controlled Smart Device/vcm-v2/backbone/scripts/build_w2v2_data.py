#!/usr/bin/env python3
"""Prepare raw-audio + transcript data for the ASR backbone fine-tune.

The active backbone is Whisper base.en (backbone/scripts/finetune_whisper.py);
this same JSONL format was also used by the abandoned wav2vec2 CTC attempt
(archive/w2v2_base/). Whisper consumes RAW 16 kHz waveforms. This script
builds two JSONL files:

    backbone/data/train.jsonl   ME2 positives (train split)
    backbone/data/val.jsonl     ME2 positives (val split)

Each row: {id, audio (abs path), text (spoken transcript), intent, slots}

LEAKAGE-FREE (canonical): run with --no-newspk so that NO
additional_test_data clip (the 171 raw clips AND every derived
denoised/noise/reverb/pitch variant) enters train or val. The 171 raw
clips then remain a genuinely unseen held-out test set. This is the
configuration the shipped model was trained on:
    python backbone/scripts/build_w2v2_data.py --no-newspk
    -> train=37992  val=5315  (ME2 optionb positives only)

Without --no-newspk the 8-variant new-speaker subset (1368 clips) is added
to train and the 171 clean clips to val -- a CONTAMINATED configuration
that was used only for an early (discarded) experiment. Do not use it.
"""
import argparse
import csv
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../backbone/scripts
_BACK = os.path.dirname(_HERE)                              # .../backbone
_REPO = os.path.dirname(_BACK)                              # .../VCM-v2
_SANDBOX = os.path.dirname(os.path.dirname(_ME2))  # .../sandbox
_ME2 = os.path.dirname(_REPO)  # vcm-v2 lives inside the ME2 folder
sys.path.insert(0, _REPO)
sys.path.insert(0, _ME2)

from model.train import tokenize  # noqa: E402  (exact ME2 training tokenizer)

ME2_MANIFEST = os.path.join(_ME2, "data", "manifests",
                            "positive_negative_manifest.csv")
ME2_DATA_ROOT = os.path.join(_ME2, "data")
SUBSET_MANIFEST = os.path.join(_REPO, "archive", "ctc_v8", "data_new",
                               "subset_manifest.csv")
OUT_DIR = os.path.join(_BACK, "data")

# Same OOV target fixes as the CTC subset build (see archive build_subset.py).
TARGET_FIX = {
    "end playback": "stop playing",
    "voluyme up": "volume up",
}


def _tok_ok(text):
    """Return the token id list, or None if it tokenizes to empty."""
    t = tokenize(text)
    return t if t else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-newspk", action="store_true",
                    help="exclude ALL additional_test_data clips "
                         "(variants AND clean) from train+val so the 171 "
                         "raw clips stay a leakage-free held-out test set")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- ME2 positives (train + val splits) ----
    me2_rows = {"train": [], "val": []}
    with open(ME2_MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["polarity"] != "positive":
                continue
            split = r["split"]
            if split not in me2_rows:
                continue
            audio = r["audio"]
            if not os.path.isabs(audio):
                audio = os.path.join(ME2_DATA_ROOT, audio)
            text = r["transcript"].strip()
            if not text or not os.path.isfile(audio):
                continue
            toks = _tok_ok(text)
            if not toks:
                continue
            me2_rows[split].append({
                "id": r["id"], "audio": audio, "text": text,
                "intent": r["intent"], "slots": json.loads(r["slots"]),
            })
    print(f"ME2 positives: train={len(me2_rows['train'])} "
          f"val={len(me2_rows['val'])}")

    # ---- New-speaker subset (8 variants -> train, clean -> val) ----
    # With --no-newspk this is skipped entirely: the 171 raw additional_test
    # clips (and every derived variant) stay OUT of train+val so the held-out
    # test set is genuinely unseen.
    sub_train, sub_val = [], []
    if args.no_newspk:
        print("New-speaker subset: SKIPPED (--no-newspk)")
    else:
        with open(SUBSET_MANIFEST) as f:
            for r in csv.DictReader(f):
                variant = r["variant"]
                text = TARGET_FIX.get(r["spoken"], r["spoken"])
                toks = _tok_ok(text)
                if not toks:
                    continue
                audio = r["path"]
                # The manifest was written before the archive move; remap.
                if not os.path.isfile(audio):
                    audio = audio.replace("/data_new/",
                                          "/archive/ctc_v8/data_new/")
                row = {
                    "id": f"newspk_{r['source']}_{variant}",
                    "audio": audio, "text": text,
                    "intent": r["folder"], "slots": {},
                }
                if variant == "clean":
                    sub_val.append(row)
                else:
                    sub_train.append(row)
    print(f"New-speaker subset: train={len(sub_train)} val={len(sub_val)}")

    # ---- Write ----
    for name, rows in [("train", me2_rows["train"] + sub_train),
                       ("val", me2_rows["val"] + sub_val)]:
        p = os.path.join(OUT_DIR, f"{name}.jsonl")
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"wrote {p}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
