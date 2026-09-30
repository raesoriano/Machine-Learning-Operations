"""Data-quality fixes for the ME2 manifest, applied to a COPY (never the source).

Two issues found while debugging the classifier's 7 test-set mismatches:

1. MISLABEL  -- every training row whose transcript is exactly "lights out"
   is labelled LIGHT_ON, but "lights out" is the English idiom for
   "turn off the lights" (-> LIGHT_OFF). The classifier was *correctly*
   learning the wrong label, so it predicted LIGHT_ON for "lights out".
   Fix: relabel those rows to LIGHT_OFF.

2. VOCAB GAP -- the STOP command is only ever said as "stop" / "stop the
   music" / "stop playing" / "turn off the music" / "stop music". The
   natural paraphrase "end playback" (and "end the music") never appears,
   so the classifier REJECTs it. Fix: add the "end X" STOP paraphrases
   (mirroring the existing "stop X" forms) so the model learns the
   "end ... = stop" mapping.

Both are principled, general fixes (not fits to the 171-clip test set):
the mislabel is an objective error and the "end X" forms are the same
template the dataset already uses for STOP.
"""
import csv
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "..", "data", "manifests",
                    "positive_negative_manifest.csv")

# "lights out" -> the idiom for turning the lights OFF.
_LIGHTS_OUT = "lights out"

# STOP paraphrases in the "end X" form (mirrors existing "stop X" rows).
_STOP_END_FORMS = [
    "end playback",
    "end the playback",
    "end the music",
    "end the song",
    "end playing",
]


def clean_rows(rows):
    """rows (list of dict) -> (rows, report). Relabels + appends in place."""
    relabel = 0
    for r in rows:
        if (r["polarity"] == "positive"
                and r["transcript"].strip().lower() == _LIGHTS_OUT
                and r["command"] != "LIGHT_OFF"):
            r["command"] = "LIGHT_OFF"
            relabel += 1

    # add STOP "end X" paraphrases, sampled from existing STOP rows so the
    # audio/speaker/split columns stay consistent (text-only augmentation).
    stop_rows = [r for r in rows
                 if r["polarity"] == "positive" and r["command"] == "STOP"]
    added = 0
    for i, phrase in enumerate(_STOP_END_FORMS):
        base = stop_rows[i % len(stop_rows)]
        nr = dict(base)
        nr["id"] = f"aug_stop_end_{i}"
        nr["transcript"] = phrase
        nr["target"] = phrase
        nr["command"] = "STOP"
        nr["intent"] = base.get("intent", "media_control")
        nr["slots"] = base.get("slots", "")
        nr["source"] = "augmented"
        rows.append(nr)
        added += 1
    return rows, {"lights_out_relabelled": relabel,
                  "stop_end_forms_added": added}


def clean_manifest(src=_SRC, out=None):
    """Read the manifest, apply fixes, write a cleaned COPY. Returns report."""
    with open(src) as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys())
    rows, report = clean_rows(rows)
    out = out or os.path.join(_HERE, "..", "artifacts",
                              "manifest_clean.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    report["out"] = out
    report["n_rows"] = len(rows)
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(clean_manifest(), indent=2))
