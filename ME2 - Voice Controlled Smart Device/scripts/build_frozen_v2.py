#!/usr/bin/env python3
"""Build frozen_test_v2.jsonl — the command-level frozen benchmark set.

Plan A benchmark: score at the COMMAND level (one of the 31 commands, or
reject). So each row carries the gold `command` (in-domain) / reject label
(OOD), plus the `transcript` (for WER) and the audio path.

  * in_domain rows: manifest test-split POSITIVES -> gold = `command`
  * ood_reject rows: manifest test-split NEGATIVES -> gold = reject

The set is frozen once built (never regenerated); v3 would be a new file.
"""
import csv, json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN = os.path.join(REPO, "data", "manifests",
                   "positive_negative_manifest.csv")
OUT = os.path.join(REPO, "data", "manifests", "frozen_test_v2.jsonl")

with open(MAN) as f:
    rows = [r for r in csv.DictReader(f) if r["split"] == "test"]

out_rows = []
for r in rows:
    if r["polarity"] == "positive":
        out_rows.append({
            "id": r["id"], "audio": r["audio"], "command": r["command"],
            "intent": r["intent"], "transcript": r["transcript"],
            "subset": "in_domain",
        })
    else:
        out_rows.append({
            "id": r["id"], "audio": r["audio"], "command": "REJECT",
            "intent": "unknown", "transcript": "",
            "subset": "ood_reject",
        })

with open(OUT, "w") as f:
    for r in out_rows:
        f.write(json.dumps(r) + "\n")

from collections import Counter
print(f"wrote {len(out_rows)} rows -> {OUT}")
print("subsets:", dict(Counter(r["subset"] for r in out_rows)))
print("intents:", dict(Counter(r["intent"] for r in out_rows)))
print("commands:", len({r["command"] for r in out_rows if r["subset"] == "in_domain"}))
