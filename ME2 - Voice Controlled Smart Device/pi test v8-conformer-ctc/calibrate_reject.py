#!/usr/bin/env python3
"""pi test v8 -- calibrate the reject rule on a merged eval report.

The expanded-vocab model DECODES out-of-scope speech to real words (the free
decode is no longer empty), so the old `--reject-empty` rule is wrong for it.
Rejection must come from a SCORE rule. This script sweeps candidate rules on
the per-clip scores saved by eval_v8.py (c_lp, f_lp, gap, c_score, f_score)
and reports, for each rule:
  * reject_acc      : OOS clips correctly rejected
  * false-reject    : in-scope clips wrongly rejected (real / synthetic)
  * overall_acc     : everything together
  * command_acc     : in-scope accuracy

Rules swept (all thresholds on per-frame log-prob units):
  A  gap > m                       (free decodes far better than the grammar)
  B  c_score < t                   (best grammar phrase scores poorly)
  C  gap > m  AND  c_score < t
  D  (free empty) OR gap > m       (keep the empty-decode safety net)

Usage:
    python calibrate_reject.py --report _eval_v2_test.json
"""
from __future__ import annotations
import argparse
import json


def eval_rule(results, pred_fn):
    inscope = [r for r in results if r["gold"] != "REJECT"]
    oos = [r for r in results if r["gold"] == "REJECT"]
    rej = [r for r in results if pred_fn(r)]
    rej_set = {id(r) for r in rej}
    rej_oos = sum(1 for r in oos if id(r) in rej_set)
    fr_real = [r for r in inscope if id(r) in rej_set and not r["is_syn"]]
    fr_syn = [r for r in inscope if id(r) in rej_set and r["is_syn"]]
    correct = 0
    for r in results:
        p = "REJECT" if id(r) in rej_set else r["pred"]
        correct += p == r["gold"]
    return {
        "reject_acc": rej_oos / max(1, len(oos)),
        "n_rej_oos": rej_oos,
        "fr_real": len(fr_real) / max(1, len(fr_real) + len([r for r in inscope if not r["is_syn"]])),
        "fr_syn": len(fr_syn) / max(1, len(fr_syn) + len([r for r in inscope if r["is_syn"]])),
        "fr_n": len(fr_real) + len(fr_syn),
        "overall_acc": correct / max(1, len(results)),
        "cmd_acc": sum(1 for r in inscope if r["pred"] == r["gold"]
                       and id(r) not in rej_set) / max(1, len(inscope)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--gap-grid", default="0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6,0.8,1.0")
    ap.add_argument("--cscore-grid", default="-6,-5,-4.5,-4,-3.5,-3,-2.5,-2,-1.5,-1,-0.5,0")
    args = ap.parse_args()

    d = json.load(open(args.report))
    results = d["results"]
    n_oos = sum(1 for r in results if r["gold"] == "REJECT")
    n_ins = len(results) - n_oos
    print(f"report: {args.report}  n={len(results)}  in-scope={n_ins}  OOS={n_oos}")

    gaps = [float(x) for x in args.gap_grid.split(",")]
    cs = [float(x) for x in args.cscore_grid.split(",")]

    print("\n=== Rule A: reject if gap > m ===")
    print(f"{'m':>6} {'rej_acc':>8} {'FR real':>8} {'FR syn':>8} {'FR n':>5} {'overall':>8} {'cmd':>8}")
    for m in gaps:
        s = eval_rule(results, lambda r, m=m: r["gap"] > m)
        print(f"{m:>6.2f} {s['reject_acc']:>8.4f} {s['fr_real']:>8.4f} "
              f"{s['fr_syn']:>8.4f} {s['fr_n']:>5d} {s['overall_acc']:>8.4f} {s['cmd_acc']:>8.4f}")

    print("\n=== Rule B: reject if c_score < t ===")
    print(f"{'t':>6} {'rej_acc':>8} {'FR real':>8} {'FR syn':>8} {'FR n':>5} {'overall':>8} {'cmd':>8}")
    for t in cs:
        s = eval_rule(results, lambda r, t=t: r["c_score"] < t)
        print(f"{t:>6.2f} {s['reject_acc']:>8.4f} {s['fr_real']:>8.4f} "
              f"{s['fr_syn']:>8.4f} {s['fr_n']:>5d} {s['overall_acc']:>8.4f} {s['cmd_acc']:>8.4f}")

    print("\n=== Rule C: reject if gap > m AND c_score < t ===")
    print(f"{'m':>6} {'t':>6} {'rej_acc':>8} {'FR real':>8} {'FR syn':>8} {'FR n':>5} {'overall':>8} {'cmd':>8}")
    best = None
    for m in gaps:
        for t in cs:
            s = eval_rule(results, lambda r, m=m, t=t: r["gap"] > m and r["c_score"] < t)
            # score: reward reject acc, penalize false rejects
            score = s["reject_acc"] - 0.5 * (s["fr_real"] + s["fr_syn"])
            if best is None or score > best[0]:
                best = (score, m, t, s)
            print(f"{m:>6.2f} {t:>6.2f} {s['reject_acc']:>8.4f} {s['fr_real']:>8.4f} "
                  f"{s['fr_syn']:>8.4f} {s['fr_n']:>5d} {s['overall_acc']:>8.4f} {s['cmd_acc']:>8.4f}")

    print("\n=== Rule D: reject if (free empty) OR gap > m ===")
    print(f"{'m':>6} {'rej_acc':>8} {'FR real':>8} {'FR syn':>8} {'FR n':>5} {'overall':>8} {'cmd':>8}")
    for m in gaps:
        s = eval_rule(results, lambda r, m=m: (not r["free"]) or r["gap"] > m)
        print(f"{m:>6.2f} {s['reject_acc']:>8.4f} {s['fr_real']:>8.4f} "
              f"{s['fr_syn']:>8.4f} {s['fr_n']:>5d} {s['overall_acc']:>8.4f} {s['cmd_acc']:>8.4f}")

    if best:
        score, m, t, s = best
        print(f"\nBEST (rule C, score=rej_acc-0.5*FR): m={m} t={t}  "
              f"rej_acc={s['reject_acc']:.4f} FR real={s['fr_real']:.4f} "
              f"FR syn={s['fr_syn']:.4f} overall={s['overall_acc']:.4f} cmd={s['cmd_acc']:.4f}")


if __name__ == "__main__":
    main()
