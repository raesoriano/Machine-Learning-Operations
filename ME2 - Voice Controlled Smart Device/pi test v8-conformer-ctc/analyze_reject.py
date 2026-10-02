#!/usr/bin/env python3
"""Discriminator hunt v2: FSA-locked-short-command + combined rules.

Idea: for a long OOS utterance the permissive FSA locks onto a SHORT command
(e.g. 'time', 'lights'), so phrase_len << free_len. For in-scope the
constrained phrase ~= free decode length.
"""
import json
import numpy as np

d = json.load(open("_eval_v2b_test.json"))
R = d["results"]
words = [w.strip() for w in open("models_v2/words.txt") if w.strip()]
base = set(words[:109])

ins = [r for r in R if r["gold"] != "REJECT"]
oos = [r for r in R if r["gold"] == "REJECT"]
ins_real = [r for r in ins if not r["is_syn"]]
ins_syn = [r for r in ins if r["is_syn"]]

for r in R:
    fw = r["free"].split()
    pw = (r.get("phrase") or "").split()
    r["bf"] = (sum(1 for w in fw if w in base) / len(fw)) if fw else 0.0
    r["free_len"] = len(fw)
    r["phrase_len"] = len(pw)
    r["plen_ratio"] = (len(pw) / len(fw)) if fw else 0.0   # low => OOS


def show(name, grp):
    a = np.array([r[name] for r in grp])
    print(f"{name:12s} {grp[0]['gold'][:4]:4s} n={len(a):4d}  "
          f"p5={np.percentile(a,5):6.2f} p10={np.percentile(a,10):6.2f} "
          f"med={np.median(a):6.2f} p90={np.percentile(a,90):6.2f}")

print("=== per-clip discriminators ===")
for nm in ["bf", "free_len", "phrase_len", "plen_ratio"]:
    for grp in [ins_real, ins_syn, oos]:
        show(nm, grp)
    print()


def score_rule(rejset):
    rej_oos = sum(1 for r in oos if id(r) in rejset)
    fr_r = sum(1 for r in ins_real if id(r) in rejset)
    fr_s = sum(1 for r in ins_syn if id(r) in rejset)
    correct = sum(1 for r in R if ("REJECT" if id(r) in rejset else r["pred"]) == r["gold"])
    cmd = sum(1 for r in ins if r["pred"] == r["gold"] and id(r) not in rejset)
    ra = rej_oos / len(oos)
    fr = fr_r / len(ins_real)
    fs = fr_s / len(ins_syn)
    return ra, fr, fs, correct / len(R), cmd / len(ins), \
        ra - 2.0 * fr - 0.5 * fs


def sweep(name, rule, grid):
    print(f"=== {name} ===")
    print(f"{'thr':>6} {'rej':>6} {'FRreal':>7} {'FRsyn':>6} {'overall':>8} {'cmd':>7} {'score':>7}")
    best = None
    for thr in grid:
        rejset = {id(r) for r in R if rule(r, thr)}
        ra, fr, fs, ov, ca, sc = score_rule(rejset)
        if best is None or sc > best[0]:
            best = (sc, thr, ra, fr, fs, ov, ca)
        print(f"{thr:>6.2f} {ra:>6.3f} {fr:>7.3f} {fs:>6.3f} {ov:>8.4f} {ca:>7.4f} {sc:>7.3f}")
    print(f"BEST: thr={best[1]:.2f}  rej={best[2]:.3f}  FRr={best[3]:.3f}  "
          f"FRs={best[4]:.3f}  overall={best[5]:.4f}  cmd={best[6]:.4f}  score={best[0]:.3f}\n")
    return best


# rule: reject if (base_frac < bf_t) OR (free_len>=2 AND plen_ratio < pr_t)
print("=== COMBO: (bf<0.6) OR (free_len>=2 AND plen_ratio<pr) ===")
print(f"{'pr':>6} {'rej':>6} {'FRreal':>7} {'FRsyn':>6} {'overall':>8} {'cmd':>7} {'score':>7}")
best = None
for pr in np.arange(0.0, 0.901, 0.05):
    def rule(r, pr):
        if r["bf"] < 0.6:
            return True
        if r["free_len"] >= 2 and r["plen_ratio"] < pr:
            return True
        return False
    rejset = {id(r) for r in R if rule(r, pr)}
    ra, fr, fs, ov, ca, sc = score_rule(rejset)
    if best is None or sc > best[0]:
        best = (sc, pr, ra, fr, fs, ov, ca)
    print(f"{pr:>6.2f} {ra:>6.3f} {fr:>7.3f} {fs:>6.3f} {ov:>8.4f} {ca:>7.4f} {sc:>7.3f}")
print(f"BEST COMBO: pr={best[1]:.2f}  rej={best[2]:.3f}  FRr={best[3]:.3f}  "
      f"FRs={best[4]:.3f}  overall={best[5]:.4f}  cmd={best[6]:.4f}  score={best[0]:.3f}\n")

# reference old
old = json.load(open("_eval_v2_test.json"))
oldR = {r["file"]: r for r in old["results"]}
print("=== reference OLD (109-vocab, reject-empty): OOS rej / real FR / synth FR ===")
for name, grp in [("OOS", oos), ("real", ins_real), ("synth", ins_syn)]:
    n = len(grp)
    old_rej = sum(1 for r in grp if oldR[r["file"]]["pred"] == "REJECT")
    print(f"{name:6s} n={n:4d}  old reject={old_rej/n:.3f}")
