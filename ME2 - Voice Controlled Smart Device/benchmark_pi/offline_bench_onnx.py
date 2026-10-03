#!/usr/bin/env python3
"""ME2 offline benchmark -- ONNX runner (Raspberry Pi edition, no torch).

Same 202-clip class holdout set and the SAME unmodified scoring code as
../benchmark (vcmbench), but the v8new model is decoded with the ONNX
runtime instead of torch:

    pi test v8-conformer-ctc/v8_onnx.py   (V8Onnx: log-mel front-end and
    the 93-phrase CTC-FSA are baked into the ONNX graph + meta.json)

The decode is byte-for-byte the same protocol as V8Onnx.decode() --
constrained CTC-FSA decode + greedy free decode + content-based reject
rule -- except the wav file read is skipped (audio comes from the
holdout parquet in memory), so the measured latency is pure
ONNX-inference + decode, matching what the live demo reports.

Dependencies: numpy, pandas, pyarrow, soundfile, onnxruntime. NO torch.

Usage:
    python offline_bench_onnx.py            # runs v8new_onnx
    python make_report.py                   # renders results/v8new_onnx_report.md

Outputs (in this folder's results/):
    v8new_onnx_metrics.json   all metrics (intent/command accuracy, false
                              accept/reject, misfire, slot exact, latency)
    v8new_onnx_trials.csv     per-clip: true vs predicted, latency
    v8new_onnx_report.md      benchmark-format report
    summary.json
"""
from __future__ import annotations
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ME2 = HERE.parent
BENCH = ME2 / "benchmark"          # sibling folder: holdout loader + scoring
sys.path.insert(0, str(BENCH))

# Reuse the torch benchmark's holdout loader, fine-class mapping, and the
# unmodified vcmbench scoring. offline_bench.py imports torch only INSIDE
# its run_v8(), so importing it here is torch-free.
from offline_bench import (          # noqa: E402
    _jsonable, load_holdout, map_fine, score_model,
)

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def run_v8new_onnx(clips, model_dir=None, reject_margin=8.0):
    """Decode each clip with the ONNX runtime (v8_onnx.py protocol)."""
    v8dir = ME2 / "pi test v8-conformer-ctc"
    v8 = _load_module("bench_v8onnx", str(v8dir / "v8_onnx.py"))
    onnx_path = Path(model_dir or (v8dir / "models")) / "best.onnx"
    rec = v8.V8Onnx(str(onnx_path), reject_margin=reject_margin,
                    reject_content=True, base_vocab_size=109)
    print(f"  onnx: {onnx_path}  vocab={len(rec.words)} "
          f"phrases={len(rec.fsa.phrases)}", flush=True)
    out = []
    for c in clips:
        x = c["audio"]                      # float32 mono 16 kHz, in memory
        t0 = time.perf_counter()
        lp = rec._logprobs(x)               # [T', V+1] log-probs
        c_lp, c_words = rec.fsa.decode(lp)  # constrained CTC-FSA
        f_lp, f_words = v8.greedy_ctc(lp)   # free decode
        ms = (time.perf_counter() - t0) * 1000.0
        # --- same reject protocol as V8Onnx.decode() ---
        if c_words is not None:
            phrase = " ".join(rec.words[w - 1] for w in c_words)
            fine = v8.phrase_to_class(phrase)
        else:
            phrase, fine = "", "REJECT"
        if (f_lp - c_lp) / max(1, lp.shape[0]) > rec.reject_margin:
            fine = "REJECT"
        if rec.reject_content:
            fw = [w - 1 for w in f_words]
            pw = [w - 1 for w in c_words] if c_words else []
            base_frac = (sum(1 for w in fw if w < rec.base_vocab_size)
                         / len(fw)) if fw else 0.0
            plen_ratio = (len(pw) / len(fw)) if fw else 0.0
            if (base_frac < 0.6) or (len(fw) >= 2 and plen_ratio < 0.20):
                fine = "REJECT"
        out.append((fine, ms))
    return out

RUNNERS = {
    "v8new_onnx": lambda clips: run_v8new_onnx(clips),
}

def main():
    only = sys.argv[1:] or ["v8new_onnx"]
    clips = load_holdout()
    n_oos = sum(1 for c in clips if c["true_intent"] == "OUT_OF_SCOPE")
    print(f"holdout: {len(clips)} clips ({len(clips)-n_oos} in-scope, "
          f"{n_oos} OOS), {sum(c['is_synthetic'] for c in clips)} synthetic",
          flush=True)
    outdir = HERE / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name in only:
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        preds = RUNNERS[name](clips)
        dt = time.time() - t0
        print(f"  decoded {len(preds)} clips in {dt:.1f}s", flush=True)
        res, trials = score_model(name, clips, preds)
        p = outdir / f"{name}_metrics.json"
        p.write_text(json.dumps(_jsonable(res), indent=2))
        with open(outdir / f"{name}_trials.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "file", "true_intent", "true_variation",
                        "true_slot", "pred_intent", "pred_slot",
                        "pred_variation", "is_synthetic", "infer_ms",
                        "correct_intent", "correct_command"])
            for c, t in zip(clips, trials):
                w.writerow([c["idx"], c["file"], t["true_intent"],
                            t["true_variation"], t["true_slot"],
                            t["pred_intent"], t["pred_slot"],
                            t["pred_variation"], int(t["is_synthetic"]),
                            t["infer_ms"], t.get("correct_intent"),
                            t.get("correct_command")])
        ov = res["breakdowns"]["overall"]
        summary[name] = {
            "n": ov["n"],
            "intent_acc": round(ov["accuracy"], 4),
            "intent_acc_ci95": [round(x, 4) for x in ov["accuracy_ci95"]],
            "intent_balanced": round(ov["balanced_accuracy"], 4),
            "intent_macro_f1": round(ov["macro_f1"], 4),
            "intent_macro_f2": round(ov["macro_f2"], 4),
            "false_accept": round(ov["false_accept_rate"], 4),
            "false_reject": round(ov["false_reject_rate"], 4),
            "misfire": round(ov["misfire_rate"], 4),
            "cmd_acc": round(ov["command_accuracy"], 4),
            "cmd_balanced": round(ov["command_balanced_accuracy"], 4),
            "cmd_macro_f1": round(ov["command_macro_f1"], 4),
            "cmd_macro_f2": round(ov["command_macro_f2"], 4),
            "cmd_misfire": round(ov["command_misfire_rate"], 4),
            "slot_exact": (round(ov["slot_exact_rate"], 4)
                           if ov.get("slot_exact_rate") is not None else None),
            "latency_p50_ms": round((ov.get("latency_p50") or 0) * 1000, 1),
            "latency_p95_ms": round((ov.get("latency_p95") or 0) * 1000, 1),
            "real_intent_acc": round(res["breakdowns"]["real voice"]["accuracy"], 4),
            "synth_intent_acc": round(res["breakdowns"]["synthetic voice"]["accuracy"], 4),
        }
        print(f"  -> {p.name}  intent_acc={summary[name]['intent_acc']} "
              f"cmd_acc={summary[name]['cmd_acc']} "
              f"FA={summary[name]['false_accept']} "
              f"FR={summary[name]['false_reject']} "
              f"misfire={summary[name]['misfire']}", flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
