#!/usr/bin/env python3
"""pi test v6 -- evaluate the HMM/GMM recognizer on the AI231 ME2 dataset.

Reads the per-split manifest.csv (the new HuggingFace set) and scores the
two-level decoder:

    waveform -> features
             -> constrained decode (93-phrase FSA x phone HMMs) -> phrase
             -> free decode (unigram) -> "what I actually heard"
    REJECT if the grammar decode is empty or the free decode is far better
    (the utterance is not one of our commands).

Gold label = the manifest `command` column (the coarse 19-command schema).
OUT_OF_SCOPE rows are the REJECT class. The recognizer emits a fine slot
class; it is mapped to the coarse schema via commands.coarse_class before
comparison, so a clip is correct when the coarse command matches.

Report: overall / in-scope / reject accuracy, intent accuracy, per-class
breakdown, mean WER (constrained vs free transcript), decode latency,
real-vs-synthetic split (test is ~77% synthetic, so both are reported).

Parallel: one decoder per worker process (ProcessPoolExecutor).
"""
from __future__ import annotations
import os, sys, json, time, csv, argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hgm.feats import read_wav, features
from hgm.acoustic import AcousticModel
from hgm.dict import load as load_dict
from hgm.lm import LanguageModel
from hgm.decode import Decoder
from hgm.commands import phrase_to_class, class_to_intent, coarse_class

DEFAULT_DATA = "/home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset"
MODELS = os.path.join(HERE, "models")

# coarse 19-command schema -> intent (matches the fine->intent map in commands.py)
COARSE_TO_INTENT = {
    "ALARM": "set_alarm",
    "BRIGHTNESS": "lights_adjust", "COLOR": "lights_adjust",
    "TEMPERATURE": "set_temperature",
    "CREATE_REMINDER": "reminders_lists", "LIST_REMINDERS": "reminders_lists",
    "TIMER": "set_timer",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIGHT_OFF": "lights_switch", "LIGHT_ON": "lights_switch",
    "PLAY_MUSIC": "play_music", "NEXT": "media_control", "PAUSE": "media_control",
    "STOP": "media_control",
    "TIME": "ask_question", "WEATHER": "ask_question",
    "CALL": "call", "MESSAGE": "call",
    "REJECT": "unknown",
}

# per-worker decoder, built once in each process
_W = {}


def _init_worker(models_dir: str, beam: int):
    am = AcousticModel.load(os.path.join(models_dir, "acoustic_model.npz"))
    dictionary = load_dict(os.path.join(models_dir, "dictionary.txt"))
    lm = LanguageModel.load(os.path.join(models_dir, "lm.npz"))
    _W["dec"] = Decoder(am, dictionary, lm.words, beam=beam)
    _W["lm"] = lm


def _wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    n, m = len(r), len(h)
    dp = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        dp[i, 0] = i
    for j in range(m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return dp[n, m] / max(1, n)


def _eval_one(args):
    path, gold, is_syn, margin = args
    dec, lm = _W["dec"], _W["lm"]
    x = read_wav(path)
    F = features(x)
    T = F.shape[0]
    t1 = time.perf_counter()
    words, glp, _ = dec.decode_constrained(lm, F)
    fwords, flp = dec.decode_free(lm, F)
    asr_ms = (time.perf_counter() - t1) * 1000
    glp_pf = glp / max(1, T)
    flp_pf = flp / max(1, T)
    phrase = " ".join(words)
    fine = phrase_to_class(phrase) if words else "REJECT"
    pred = coarse_class(fine)
    if not words or (flp_pf - glp_pf) > margin:
        pred = "REJECT"
    return {
        "path": path,
        "gold": gold, "heard": " ".join(fwords),
        "transcript": phrase, "fine": fine, "pred": pred,
        "correct": bool(pred == gold),
        "is_syn": bool(is_syn),
        "wer": _wer(phrase, " ".join(fwords)) if words else 1.0,
        "asr_ms": asr_ms,
    }


def _load_split(data: str, split: str):
    """[(wav_path, gold_coarse, is_synthetic)] from the split manifest."""
    rows = []
    mpath = os.path.join(data, split, "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            gold = "REJECT" if int(r.get("out_of_scope") or 0) == 1 \
                else r.get("command", "").strip()
            if not gold:
                continue
            p = os.path.join(data, split, r["file"])
            if os.path.exists(p):
                rows.append((p, gold, int(r.get("is_synthetic") or 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--beam", type=int, default=4000)
    ap.add_argument("--reject-margin", type=float, default=8.0,
                    help="reject if (free_lp - grammar_lp)/frames > margin")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.report is None:
        args.report = os.path.join(HERE, f"test_v6_{args.split}_report.json")

    rows = _load_split(args.data, args.split)
    print(f"{args.split} clips: {len(rows)}", flush=True)

    jobs = [(p, g, s, args.reject_margin) for (p, g, s) in rows]
    results = [None] * len(jobs)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers,
                             initializer=_init_worker,
                             initargs=(MODELS, args.beam)) as ex:
        futs = {ex.submit(_eval_one, j): i for i, j in enumerate(jobs)}
        for fut in as_completed(futs):
            i = futs[fut]
            results[i] = fut.result()
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(rows)}  ({time.time()-t0:.0f}s)", flush=True)

    n = len(results)
    inscope = [r for r in results if r["gold"] != "REJECT"]
    oos = [r for r in results if r["gold"] == "REJECT"]
    overall = sum(r["correct"] for r in results) / max(1, n)
    cmd_acc = sum(r["correct"] for r in inscope) / max(1, len(inscope))
    rej_acc = sum(r["correct"] for r in oos) / max(1, len(oos))
    intent_acc = sum(class_to_intent(r["pred"]) ==
                     COARSE_TO_INTENT.get(r["gold"], "unknown")
                     for r in results) / max(1, n)
    mean_wer = float(np.mean([r["wer"] for r in results]))
    lat = np.array([r["asr_ms"] for r in results])

    per_class = {}
    for c in sorted(set(r["gold"] for r in results)):
        sub = [r for r in results if r["gold"] == c]
        per_class[c] = {
            "n": len(sub),
            "acc": round(sum(r["correct"] for r in sub) / len(sub), 4),
        }

    def _block(sub):
        if not sub:
            return {"n": 0}
        ins = [r for r in sub if r["gold"] != "REJECT"]
        oos_ = [r for r in sub if r["gold"] == "REJECT"]
        return {
            "n": len(sub),
            "overall_acc": round(sum(r["correct"] for r in sub) / len(sub), 4),
            "command_acc": round(sum(r["correct"] for r in ins) / max(1, len(ins)), 4),
            "reject_acc": round(sum(r["correct"] for r in oos_) / max(1, len(oos_)), 4),
        }

    report = {
        "model": "v6 HMM/GMM (from-scratch PocketSphinx-arch)",
        "split": args.split, "n_clips": n,
        "n_inscope": len(inscope), "n_oos": len(oos),
        "overall_acc": round(overall, 4),
        "command_acc": round(cmd_acc, 4),
        "reject_acc": round(rej_acc, 4),
        "intent_acc": round(intent_acc, 4),
        "mean_wer": round(mean_wer, 4),
        "asr_ms_p50": round(float(np.percentile(lat, 50)), 1),
        "asr_ms_p90": round(float(np.percentile(lat, 90)), 1),
        "reject_margin": args.reject_margin,
        "real": _block([r for r in results if not r["is_syn"]]),
        "synthetic": _block([r for r in results if r["is_syn"]]),
        "per_class": per_class,
        "results": results,
    }
    json.dump(report, open(args.report, "w"), indent=1)
    print(f"\n=== {args.split} ({n} clips) ===")
    print(f"  overall_acc : {overall:.4f}")
    print(f"  command_acc : {cmd_acc:.4f}  ({len(inscope)} in-scope)")
    print(f"  reject_acc  : {rej_acc:.4f}  ({len(oos)} OOS)")
    print(f"  intent_acc  : {intent_acc:.4f}")
    print(f"  mean_wer    : {mean_wer:.4f}")
    print(f"  latency ms  : p50={report['asr_ms_p50']}  p90={report['asr_ms_p90']}")
    print(f"  REAL        : n={report['real']['n']}  overall={report['real'].get('overall_acc')}  cmd={report['real'].get('command_acc')}  rej={report['real'].get('reject_acc')}")
    print(f"  SYNTHETIC   : n={report['synthetic']['n']}  overall={report['synthetic'].get('overall_acc')}  cmd={report['synthetic'].get('command_acc')}  rej={report['synthetic'].get('reject_acc')}")
    print(f"  report      : {args.report}")


if __name__ == "__main__":
    main()
