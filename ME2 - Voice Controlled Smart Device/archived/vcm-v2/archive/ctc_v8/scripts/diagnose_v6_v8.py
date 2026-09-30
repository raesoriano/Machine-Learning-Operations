#!/usr/bin/env python3
"""Diagnose WHY me2_v8 (fine-tuned on the new-speaker subset) did not beat
me2_v6 on the held-out 171 raw clips.

Compares v6 vs v8 per-clip transcripts:
  * which clips are blank in each
  * which clips changed transcript at all
  * which clips flipped blank<->non-blank
  * command accuracy on the clips whose transcript actually changed
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from vcm2.ground_truth import build_ground_truth   # noqa: E402
from vcm2.asr import ASR                           # noqa: E402
from vcm2.classifier import load_classifier, predict  # noqa: E402
import soundfile as sf                             # noqa: E402

V6 = "Machine-Learning-Operations/ME2 - Voice Controlled Smart Device/model/checkpoints/me2_v6/best.pt"
V8 = "VCM-v2/artifacts/asr/me2_v8/best.pt"
DATA = "additional_test_data"

rows = build_ground_truth(DATA)
a6 = ASR(ckpt=os.path.abspath(V6), device="cpu")
a8 = ASR(ckpt=os.path.abspath(V8), device="cpu")
clf = load_classifier()

out = []
for r in rows:
    x, sr = sf.read(r["path"], dtype="float32")
    t6 = a6.transcribe(x, sr=sr)
    t8 = a8.transcribe(x, sr=sr)
    c6 = predict(clf, t6)[0]
    c8 = predict(clf, t8)[0]
    out.append({
        "file": os.path.basename(r["path"]), "folder": r["folder"],
        "spoken": r["spoken"], "gold": r["gold"],
        "t6": t6, "t8": t8, "c6": c6, "c8": c8,
        "changed": t6 != t8,
        "b6": not t6.strip(), "b8": not t8.strip(),
    })

n = len(out)
b6 = sum(o["b6"] for o in out)
b8 = sum(o["b8"] for o in out)
changed = [o for o in out if o["changed"]]
same = [o for o in out if not o["changed"]]
c6 = sum(o["c6"] == o["gold"] for o in out)
c8 = sum(o["c8"] == o["gold"] for o in out)
flips_b2n = [o for o in out if o["b6"] and not o["b8"]]
flips_n2b = [o for o in out if not o["b6"] and o["b8"]]

print(f"n={n}  cmd v6={c6} ({100*c6/n:.1f}%)  v8={c8} ({100*c8/n:.1f}%)")
print(f"blank v6={b6}  v8={b8}  (same set? {sum(o['b6']==o['b8'] for o in out)}/{n})")
print(f"transcript changed: {len(changed)}/{n}  unchanged: {len(same)}")
print(f"flips blank->nonblank: {len(flips_b2n)}   nonblank->blank: {len(flips_n2b)}")
print(f"\ncmd acc on CHANGED clips:  v6={sum(o['c6']==o['gold'] for o in changed)}/{len(changed)}  v8={sum(o['c8']==o['gold'] for o in changed)}/{len(changed)}")
print(f"cmd acc on UNCHANGED clips: v6={sum(o['c6']==o['gold'] for o in same)}/{len(same)}  v8={sum(o['c8']==o['gold'] for o in same)}/{len(same)}")

print("\n--- all changed clips ---")
for o in changed:
    mark6 = "OK " if o["c6"] == o["gold"] else "   "
    mark8 = "OK " if o["c8"] == o["gold"] else "   "
    print(f"  {mark6}{mark8} {o['folder']:15s} {o['spoken']!r:30s} "
          f"v6={o['t6']!r:24s} v8={o['t8']!r:24s} gold={o['gold']}")

json.dump(out, open("VCM-v2/reports/v6_v8_clip_diff.json", "w"), indent=2)
print("\n-> VCM-v2/reports/v6_v8_clip_diff.json")
