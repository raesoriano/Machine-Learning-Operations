#!/usr/bin/env python3
"""Render the benchmark-format report.md for each offline run.

Reuses the benchmark's own render_markdown() so the report matches the live
benchmark's output exactly. Fills in the meta fields the live tool would set.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vcmbench import report as R  # noqa: E402

RUNS = HERE / "results"

META = {
    "v3": "ME2 v3 - PocketSphinx ensemble (custom + stock AM, stage-2 classifier)",
    "v6": "ME2 v6 - from-scratch HMM/GMM (numpy), grammar-constrained + free decode",
    "v7": "ME2 v7 - v3 architecture, custom AM retrained on the v6 dataset",
    "v8": "ME2 v8 - causal Conformer + CTC (base, score-gap reject)",
    "v8neg": "ME2 v8 - causal Conformer + CTC (+1000 negatives, reject-empty)",
    "v8new": "ME2 v8 - causal Conformer + CTC (736-word vocab, content-based reject)",
}

def build(m: dict, name: str) -> dict:
    # score() already returns meta/intent_level/command_level/false_wake/slots/
    # breakdowns/pipeline/pi. Add the meta fields render_markdown reads.
    m.setdefault("meta", {})
    m["meta"].setdefault("student", name)
    m["meta"].setdefault("started", "offline")
    m["meta"].setdefault("wake_word", "hey rhasspy (offline: not used)")
    m["meta"].setdefault("seed", "n/a")
    m["meta"].setdefault("mode", "offline")
    m["meta"].setdefault("holdout", "airimonda/ai231-me2-voice-commands split=holdout (202 clips)")
    m["meta"].setdefault("pi_mic", "n/a")
    m["meta"].setdefault("mic_check", None)
    m["meta"]["model_note"] = META.get(name, name)
    # render_glance reads m["model"] as a dict (the model_profile); score() may
    # have left it None or a string -> normalize to {} (no FLOPs offline).
    if not isinstance(m.get("model"), dict):
        m["model"] = {}
    return m

def main():
    names = sys.argv[1:] or ["v3", "v6", "v7", "v8", "v8neg", "v8new"]
    for name in names:
        p = RUNS / f"{name}_metrics.json"
        if not p.exists():
            print(f"skip {name}: no {p.name}")
            continue
        m = build(json.loads(p.read_text()), name)
        md = R.render_markdown(m)
        # prepend a header line with the model note
        md = f"> {META.get(name, name)}\n\n" + md
        out = RUNS / f"{name}_report.md"
        out.write_text(md)
        print(f"wrote {out.name}")

if __name__ == "__main__":
    main()
