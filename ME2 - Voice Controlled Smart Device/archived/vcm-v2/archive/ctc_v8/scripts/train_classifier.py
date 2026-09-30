#!/usr/bin/env python3
r"""Train the VCM v2 Stage-2 classifier on the ME2 manifest.

Two modes:

  * TUNING (default): train on the `train` split, report accuracy on the
    HELD-OUT `val` split. This is the honest number to compare configs.
  * PRODUCTION: --split all -> train on every row (the shipped model).

  python scripts/train_classifier.py \
      --manifest <ME2>/data/manifests/positive_negative_manifest.csv

  # production model (all rows), with data-quality fixes applied:
  python scripts/train_classifier.py --manifest <...> --split all --clean
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcm2.classifier import train_classifier          # noqa: E402
from vcm2.data_clean import clean_manifest            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "classifier.pkl"))
    ap.add_argument("--split", default="train",
                    choices=["train", "val", "test", "all"],
                    help="train split to use; 'all' = production model")
    ap.add_argument("--clean", action="store_true",
                    help="apply data-quality fixes (lights-out relabel + "
                         "STOP 'end X' forms) to a copy before training")
    args = ap.parse_args()

    manifest = args.manifest
    if args.clean:
        clean = clean_manifest(src=args.manifest,
                               out=os.path.join(os.path.dirname(args.out),
                                                "manifest_clean.csv"))
        print("data clean:", {k: clean[k] for k in
                              ("lights_out_relabelled", "stop_end_forms_added")})
        manifest = clean["out"]

    split = None if args.split == "all" else args.split
    pipe, report = train_classifier(manifest, args.out, split=split)
    print(json.dumps(report, indent=2))
    with open(os.path.join(os.path.dirname(args.out),
                           "classifier_train.json"), "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
