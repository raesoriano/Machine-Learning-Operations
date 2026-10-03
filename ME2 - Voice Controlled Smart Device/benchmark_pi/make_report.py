#!/usr/bin/env python3
"""Render the benchmark-format report.md for the ONNX (Pi) run.

Same rendering as ../benchmark/make_reports.py (the benchmark's own
render_markdown), pointed at THIS folder's results/.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "benchmark"
sys.path.insert(0, str(BENCH))
from vcmbench import report as R  # noqa: E402

RUNS = HERE / "results"

META = {
    "v8new_onnx": ("ME2 v8 - causal Conformer + CTC (736-word vocab, "
                   "content-based reject) -- ONNX runtime, Raspberry Pi"),
}

def build(m: dict, name: str) -> dict:
    m.setdefault("meta", {})
    m["meta"].setdefault("student", name)
    m["meta"].setdefault("started", "offline")
    m["meta"].setdefault("wake_word", "hey rhasspy (offline: not used)")
    m["meta"].setdefault("seed", "n/a")
    m["meta"].setdefault("mode", "offline-onnx")
    m["meta"].setdefault("holdout", "airimonda/ai231-me2-voice-commands split=holdout (202 clips)")
    m["meta"].setdefault("pi_mic", "n/a")
    m["meta"].setdefault("mic_check", None)
    m["meta"]["model_note"] = META.get(name, name)
    if not isinstance(m.get("model"), dict):
        m["model"] = {}
    return m

def main():
    names = sys.argv[1:] or ["v8new_onnx"]
    for name in names:
        p = RUNS / f"{name}_metrics.json"
        if not p.exists():
            print(f"skip {name}: no {p.name}")
            continue
        m = build(json.loads(p.read_text()), name)
        md = R.render_markdown(m)
        md = f"> {META.get(name, name)}\n\n" + md
        out = RUNS / f"{name}_report.md"
        out.write_text(md)
        print(f"wrote {out.name}")

if __name__ == "__main__":
    main()
