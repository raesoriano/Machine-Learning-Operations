#!/usr/bin/env python3
"""Build a side-by-side comparison (markdown + json) of the offline benchmark
runs for the ME2 models, using each run's benchmark metrics.json."""
from __future__ import annotations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "results"

ORDER = ["v3", "v6", "v7", "v8", "v8neg", "v8new"]
LABEL = {
    "v3": "v3 (PocketSphinx ensemble)",
    "v6": "v6 (from-scratch HMM)",
    "v7": "v7 (v3 arch, AM on v6 data)",
    "v8": "v8 (Conformer+CTC, base)",
    "v8neg": "v8 + negatives (reject)",
    "v8new": "v8 (736-vocab, content reject)",
}

def load(name):
    p = RUNS / f"{name}_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    ov = m["breakdowns"]["overall"]
    real = m["breakdowns"].get("real voice", {})
    synth = m["breakdowns"].get("synthetic voice", {})
    return {
        "n": ov["n"],
        "intent_acc": ov["accuracy"],
        "intent_ci": ov["accuracy_ci95"],
        "intent_bal": ov["balanced_accuracy"],
        "intent_f1": ov["macro_f1"],
        "intent_f2": ov["macro_f2"],
        "fa": ov["false_accept_rate"],
        "fa_n": f"{ov['false_accepts']}/{ov['n_out_of_scope']}",
        "fr": ov["false_reject_rate"],
        "misfire": ov["misfire_rate"],
        "cmd_acc": ov["command_accuracy"],
        "cmd_bal": ov["command_balanced_accuracy"],
        "cmd_f1": ov["command_macro_f1"],
        "cmd_f2": ov["command_macro_f2"],
        "cmd_misfire": ov["command_misfire_rate"],
        "slot": ov.get("slot_exact_rate"),
        "lat_p50": ov.get("latency_p50"),
        "lat_p95": ov.get("latency_p95"),
        "real_intent": real.get("accuracy"),
        "synth_intent": synth.get("accuracy"),
        "real_cmd": real.get("command_accuracy"),
        "synth_cmd": synth.get("command_accuracy"),
    }

def pct(x, nd=1):
    return "-" if x is None else f"{100*x:.{nd}f}%"

def main():
    rows = {}
    for name in ORDER:
        d = load(name)
        if d:
            rows[name] = d
    names = [n for n in ORDER if n in rows]

    # markdown table
    hdr = ["metric"] + [LABEL[n] for n in names]
    def ci(c):
        return "-" if not c else f"[{100*c[0]:.0f}-{100*c[1]:.0f}%]"
    lines = ["| " + " | ".join(hdr) + " |",
             "|" + "|".join(["---"] * len(hdr)) + "|"]
    def add(label, key, fmt):
        lines.append("| " + " | ".join([label] + [fmt(rows[n][key]) for n in names]) + " |")
    add("clips", "n", lambda x: str(x))
    add("intent acc (19)", "intent_acc", lambda x: pct(x))
    lines.append("| intent acc 95% CI | " + " | ".join(ci(rows[n]["intent_ci"]) for n in names) + " |")
    add("intent balanced acc", "intent_bal", lambda x: pct(x))
    add("intent macro F1", "intent_f1", lambda x: pct(x))
    add("intent macro F2", "intent_f2", lambda x: pct(x))
    lines.append("| false accept (OOS fired) | " + " | ".join(f"{pct(rows[n]['fa'])} ({rows[n]['fa_n']})" for n in names) + " |")
    add("false reject (cmd ignored)", "fr", lambda x: pct(x))
    add("misfire (wrong cmd)", "misfire", lambda x: pct(x))
    add("command acc (93)", "cmd_acc", lambda x: pct(x))
    add("command balanced acc", "cmd_bal", lambda x: pct(x))
    add("command macro F1", "cmd_f1", lambda x: pct(x))
    add("command macro F2", "cmd_f2", lambda x: pct(x))
    add("command misfire", "cmd_misfire", lambda x: pct(x))
    add("slot exact (intent right)", "slot", lambda x: pct(x))
    add("latency p50 (s)", "lat_p50", lambda x: "-" if x is None else f"{x:.2f}")
    add("latency p95 (s)", "lat_p95", lambda x: "-" if x is None else f"{x:.2f}")
    add("real intent acc", "real_intent", lambda x: pct(x))
    add("synthetic intent acc", "synth_intent", lambda x: pct(x))
    add("real command acc", "real_cmd", lambda x: pct(x))
    add("synthetic command acc", "synth_cmd", lambda x: pct(x))

    md = "\n".join(lines)
    (RUNS / "comparison.md").write_text(md + "\n")
    (RUNS / "comparison.json").write_text(json.dumps(rows, indent=2))
    print(md)

if __name__ == "__main__":
    main()
