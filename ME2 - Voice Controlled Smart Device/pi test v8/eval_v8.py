#!/usr/bin/env python3
"""pi test v8 -- evaluate the Conformer+CTC encoder on the v6 dataset.

Same protocol as v6's eval:
  * constrained decode : CTC-FSA dynamic program over the 93 command
    phrases (uniform phrase priors, same bigram-free grammar FSA as v6)
  * free decode        : greedy CTC (collapse repeats, drop blanks)
  * REJECT             : free decodes far better than the best grammar
    phrase, i.e. (free_lp - constrained_lp)/frames > margin
  * gold               : manifest `command` column (coarse 19 schema);
    OUT_OF_SCOPE rows are the REJECT class
  * report             : overall / command / reject / intent accuracy,
    real-vs-synthetic split, per-class, latency

Run:
    python eval_v8.py --split test --report _eval_test.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "pi test v6"))
from model import V8Model                        # noqa: E402
from data import (log_mel, read_wav16, load_test,  # noqa: E402
                  build_vocab)
from hgm.spoken import tokenize_spoken           # noqa: E402
from hgm.commands import (phrase_to_class, class_to_intent,  # noqa: E402
                          coarse_class)

DEFAULT_DATA = "/home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset"
MODELS = os.path.join(_HERE, "models")

# coarse 19-command schema -> intent (same map as v6 eval)
COARSE_TO_INTENT = {
    "ALARM": "set_alarm",
    "BRIGHTNESS": "lights_adjust", "COLOR": "lights_adjust",
    "TEMPERATURE": "set_temperature",
    "CREATE_REMINDER": "reminders_lists", "LIST_REMINDERS": "reminders_lists",
    "TIMER": "set_timer",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIGHT_OFF": "lights_switch", "LIGHT_ON": "lights_switch",
    "PLAY_MUSIC": "play_music", "NEXT": "media_control",
    "PAUSE": "media_control", "STOP": "media_control",
    "TIME": "ask_question", "WEATHER": "ask_question",
    "CALL": "call", "MESSAGE": "call",
}


def _load_variations():
    """The 93 phrase variations (same source as v6 train_am)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v6_train", os.path.join(os.path.dirname(_HERE),
                                 "pi test v6", "train_am.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._load_variations(mod.VARIATIONS)


class CtcFsa:
    """CTC-FSA decoder over the command grammar.

    FSA topology (same as v6, minus the SIL regions -- CTC blanks model
    silence): start node 0, phrase nodes (p,i), end node E.
    Edge weights: uniform phrase prior on the first word, 0 on the rest
    (the grammar is a flat phrase list, like v6's FSA).

    Word indices in edges are 0-BASED (index into lp[t, 1:]).
    """

    def __init__(self, phrases, word2idx):
        self.phrases = [p.split() if isinstance(p, str) else list(p)
                        for p in phrases]
        self.start = 0
        self.end = 1
        # node id for (p, i) = 2 + p*(len+1) + i
        self.node_of = {}
        node = 2
        for p, ph in enumerate(self.phrases):
            for i in range(len(ph) + 1):
                self.node_of[(p, i)] = node
                node += 1
        self.n_nodes = node
        self.log_prior = -math.log(len(self.phrases))
        # edges: (u, v, word_idx_0based, logw); -1 = non-emitting
        self.edges = []
        for p, ph in enumerate(self.phrases):
            prev = self.start
            for i, w in enumerate(ph):
                cur = self.node_of[(p, i)]
                self.edges.append((prev, cur, word2idx[w] - 1,
                                   self.log_prior if i == 0 else 0.0))
                prev = cur
            self.edges.append((prev, self.end, -1, 0.0))
        self.in_edges = [[] for _ in range(self.n_nodes)]
        for e in self.edges:
            self.in_edges[e[1]].append(e)

    def decode(self, lp: np.ndarray):
        """lp: [T, V+1] log-posteriors (row 0 = blank).

        Returns (best_logprob, [word idx 1-based] or None).
        """
        T = lp.shape[0]
        NEG = -1e30
        # B[v]  : at node v, last frame was blank
        # NB[v] : at node v, last frame was a non-blank emission
        B = np.full(self.n_nodes, NEG)
        NB = np.full(self.n_nodes, NEG)
        B[0] = 0.0
        bp_B = np.zeros((T, self.n_nodes), dtype=np.int32)
        bp_NB = np.zeros((T, self.n_nodes), dtype=np.int32)
        bp_w = np.full((T, self.n_nodes), -1, dtype=np.int32)
        # track which state (0=B, 1=NB) was best at each (t, v)
        state = np.zeros((T, self.n_nodes), dtype=np.int8)
        blank = lp[:, 0]
        for t in range(T):
            wlog = lp[t, 1:]
            nB = np.maximum(B, NB) + blank[t]
            bp_B[t] = np.arange(self.n_nodes)
            nNB = np.full(self.n_nodes, NEG)
            for v, ins in enumerate(self.in_edges):
                best = NEG
                bu = bw = -1
                for (u, _, w, lw) in ins:
                    if w < 0:
                        val = max(B[u], NB[u]) + lw
                    else:
                        val = max(B[u], NB[u]) + lw + wlog[w]
                    if val > best:
                        best = val
                        bu, bw = u, w
                if best > NEG:
                    nNB[v] = best
                    bp_NB[t][v] = bu
                    bp_w[t][v] = bw
            # record which state wins at each node
            state[t] = (nNB > nB).astype(np.int8)
            B, NB = nB, nNB
        final = max(B[self.end], NB[self.end])
        if final <= NEG / 2:
            return NEG, None
        # backtrace
        words = []
        v = self.end
        s = 1 if NB[self.end] > B[self.end] else 0
        for t in range(T - 1, -1, -1):
            if s == 1:  # non-blank emission at t
                u = int(bp_NB[t][v])
                w = int(bp_w[t][v])
                if w >= 0:
                    words.append(w + 1)   # back to 1-based
                v_prev = u
            else:      # blank at t -> same node
                v_prev = v
            if t == 0:
                break
            v = v_prev
            s = int(state[t - 1][v])
        words.reverse()
        return final, words


def greedy_ctc(lp: np.ndarray):
    """Greedy CTC decode. Returns (logprob, [word idx])."""
    pred = lp.argmax(axis=1)
    words, prev = [], -1
    for p in pred:
        if p != prev and p != 0:
            words.append(int(p))
        prev = int(p)
    # log-prob of the greedy path (blanks + emitted words)
    lp_path = 0.0
    prev = -1
    for t, p in enumerate(pred):
        p = int(p)
        if p != 0:
            if p != prev:
                lp_path += lp[t, p]
        else:
            lp_path += lp[t, 0]
        prev = p
    return lp_path, words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--model", default=os.path.join(MODELS, "best.pt"))
    ap.add_argument("--reject-margin", type=float, default=8.0,
                    help="reject if (free_lp - constrained_lp)/frames > margin")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    if args.report is None:
        args.report = os.path.join(_HERE, f"test_v8_{args.split}_report.json")

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.model, map_location=dev)
    words = ck["words"]
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    model = V8Model(ck["n_words"], d_model=ck["d_model"],
                    layers=ck["layers"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M "
          f"params  vocab={ck['n_words']}", flush=True)

    rows = _load_variations()
    phrases = [" ".join(ws) for _, _, _, _, ws in rows]
    fsa = CtcFsa(phrases, word2idx)

    rows = load_test(args.data, args.split)
    print(f"{args.split} clips: {len(rows)}", flush=True)

    results = []
    t0 = time.time()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        mels, lens = [], []
        for (p, _, _) in chunk:
            x = read_wav16(p)
            m = log_mel(x)
            if m.shape[0] > 800:
                m = m[:800]
            mels.append(m)
            lens.append(m.shape[0])
        T = max(lens)
        mel = torch.full((len(chunk), T, 80),
                         math.log(1e-5), device=dev)
        for j, m in enumerate(mels):
            mel[j, :m.shape[0], :] = m.to(dev)
        t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(mel)                       # [B, T', V+1] logprobs
        enc_ms = (time.perf_counter() - t1) * 1000.0
        out = out.float().cpu().numpy()
        for j, ((p, gold, is_syn), L) in enumerate(zip(chunk, lens)):
            Tj = (L - 2) // 4 + 1
            lp = out[j, :Tj, :]
            t2 = time.perf_counter()
            c_lp, c_words = fsa.decode(lp)
            f_lp, f_words = greedy_ctc(lp)
            dec_ms = (time.perf_counter() - t2) * 1000.0
            if c_words is not None:
                phrase = " ".join([words[w - 1] for w in c_words])
                fine = phrase_to_class(phrase)
                pred = coarse_class(fine)
            else:
                phrase = ""
                pred = "REJECT"
            if (f_lp - c_lp) / max(1, Tj) > args.reject_margin:
                pred = "REJECT"
            results.append({
                "file": os.path.basename(p),
                "gold": "REJECT" if gold == "OUT_OF_SCOPE" else gold,
                "pred": pred,
                "phrase": phrase,
                "free": " ".join([words[w - 1] for w in f_words]),
                "correct": pred == ("REJECT" if gold == "OUT_OF_SCOPE"
                                    else gold),
                "is_syn": is_syn,
                "asr_ms": round(enc_ms / len(chunk) + dec_ms, 1),
            })
        done = min(i + args.batch, len(rows))
        if done % 256 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}  ({time.time() - t0:.0f}s)",
                  flush=True)

    n = len(results)
    inscope = [r for r in results if r["gold"] != "REJECT"]
    oos = [r for r in results if r["gold"] == "REJECT"]
    overall = sum(r["correct"] for r in results) / max(1, n)
    cmd_acc = sum(r["correct"] for r in inscope) / max(1, len(inscope))
    rej_acc = sum(r["correct"] for r in oos) / max(1, len(oos))
    intent_acc = sum(COARSE_TO_INTENT.get(r["pred"], "unknown") ==
                     COARSE_TO_INTENT.get(r["gold"], "unknown")
                     for r in results) / max(1, n)
    lat = np.array([r["asr_ms"] for r in results])

    per_class = {}
    for c in sorted(set(r["gold"] for r in results)):
        sub = [r for r in results if r["gold"] == c]
        per_class[c] = {"n": len(sub),
                        "acc": round(sum(r["correct"] for r in sub)
                                     / len(sub), 4)}

    def _block(sub):
        if not sub:
            return {"n": 0}
        ins = [r for r in sub if r["gold"] != "REJECT"]
        oos_ = [r for r in sub if r["gold"] == "REJECT"]
        return {
            "n": len(sub),
            "overall_acc": round(sum(r["correct"] for r in sub) / len(sub), 4),
            "command_acc": round(sum(r["correct"] for r in ins) /
                                 max(1, len(ins)), 4),
            "reject_acc": round(sum(r["correct"] for r in oos_) /
                                max(1, len(oos_)), 4),
        }

    report = {
        "model": "v8 Conformer+CTC (causal, original template architecture)",
        "split": args.split, "n_clips": n,
        "n_inscope": len(inscope), "n_oos": len(oos),
        "overall_acc": round(overall, 4),
        "command_acc": round(cmd_acc, 4),
        "reject_acc": round(rej_acc, 4),
        "intent_acc": round(intent_acc, 4),
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
    print(f"  latency ms  : p50={report['asr_ms_p50']}  p90={report['asr_ms_p90']}")
    print(f"  REAL        : n={report['real']['n']}  overall={report['real'].get('overall_acc')}  cmd={report['real'].get('command_acc')}  rej={report['real'].get('reject_acc')}")
    print(f"  SYNTHETIC   : n={report['synthetic']['n']}  overall={report['synthetic'].get('overall_acc')}  cmd={report['synthetic'].get('command_acc')}  rej={report['synthetic'].get('reject_acc')}")
    print(f"  report      : {args.report}")


if __name__ == "__main__":
    main()
