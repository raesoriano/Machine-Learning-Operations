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

SPEED-UP (this revision):
  * sharding across GPUs: run one process per GPU, each decodes a slice of
    the clips, then merge the partial results (3 GPUs -> ~3x faster).
  * mel features are computed in a CPU thread pool while the GPU encodes
    the previous batch (overlapped, no longer serial on the main thread).

Run (single GPU, as before):
    python eval_v8.py --split test --report _eval_test.json

Run (3 GPUs, sharded -- recommended):
    python eval_v8.py --split test --num-shards 3 --shard-id 0 --device cuda:0 &
    python eval_v8.py --split test --num-shards 3 --shard-id 1 --device cuda:1 &
    python eval_v8.py --split test --num-shards 3 --shard-id 2 --device cuda:2 &
    wait
    python eval_v8.py --split test --merge --report _eval_test.json

    (each shard writes _eval_test.part{i}.json next to --report)
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import V8Model                        # noqa: E402
from data import (log_mel, read_wav16, load_test,  # noqa: E402
                  build_vocab)
from hgm.spoken import tokenize_spoken           # noqa: E402  (vendored)
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
    """The 93 phrase variations (local variations.csv, vendored from the
    dataset). Returns [(label, variation, value, phrase, spoken_words)]."""
    import csv as _csv
    rows = []
    with open(os.path.join(_HERE, "variations.csv")) as f:
        for r in _csv.DictReader(f):
            phrase = r["phrase"]
            rows.append((r["label"], r["variation"], r["value"], phrase,
                         tokenize_spoken(phrase)))
    return rows


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


def _oracle_decode(fsa, lp):
    """Pure-Python re-implementation of the same DP (no numpy arrays of
    backpointers), used ONLY by --selftest to cross-check CtcFsa.decode."""
    T = lp.shape[0]
    NEG = -1e30
    B = [NEG] * fsa.n_nodes
    NB = [NEG] * fsa.n_nodes
    B[0] = 0.0
    bp_u = [[-1] * fsa.n_nodes for _ in range(T)]
    bp_w = [[-1] * fsa.n_nodes for _ in range(T)]
    state = [[0] * fsa.n_nodes for _ in range(T)]
    for t in range(T):
        nB = [max(B[v], NB[v]) + lp[t, 0] for v in range(fsa.n_nodes)]
        nNB = [NEG] * fsa.n_nodes
        for v, ins in enumerate(fsa.in_edges):
            best, bu, bw = NEG, -1, -1
            for (u, _, w, lw) in ins:
                val = max(B[u], NB[u]) + lw + (lp[t, w + 1] if w >= 0 else 0.0)
                if val > best:
                    best, bu, bw = val, u, w
            if best > NEG:
                nNB[v] = best
                bp_u[t][v] = bu
                bp_w[t][v] = bw
        state[t] = [1 if nNB[v] > nB[v] else 0 for v in range(fsa.n_nodes)]
        B, NB = nB, nNB
    final = max(B[fsa.end], NB[fsa.end])
    if final <= NEG / 2:
        return NEG, None
    words, v, s = [], fsa.end, 1 if NB[fsa.end] > B[fsa.end] else 0
    for t in range(T - 1, -1, -1):
        if s == 1:
            u = bp_u[t][v]
            w = bp_w[t][v]
            if w >= 0:
                words.append(w + 1)
            v_prev = u
        else:
            v_prev = v
        if t == 0:
            break
        v = v_prev
        s = state[t - 1][v]
    words.reverse()
    return final, words


def build_report(results, split, args, model_tag="v8 Conformer+CTC "
               "(causal, original template architecture)"):
    """Same report structure as the original single-process eval."""
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
        "model": model_tag,
        "split": split, "n_clips": n,
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
    return report


def print_report(report):
    print(f"\n=== {report['split']} ({report['n_clips']} clips) ===")
    print(f"  overall_acc : {report['overall_acc']:.4f}")
    print(f"  command_acc : {report['command_acc']:.4f}  "
          f"({report['n_inscope']} in-scope)")
    print(f"  reject_acc  : {report['reject_acc']:.4f}  "
          f"({report['n_oos']} OOS)")
    print(f"  intent_acc  : {report['intent_acc']:.4f}")
    print(f"  latency ms  : p50={report['asr_ms_p50']}  "
          f"p90={report['asr_ms_p90']}")
    print(f"  REAL        : n={report['real']['n']}  "
          f"overall={report['real'].get('overall_acc')}  "
          f"cmd={report['real'].get('command_acc')}  "
          f"rej={report['real'].get('reject_acc')}")
    print(f"  SYNTHETIC   : n={report['synthetic']['n']}  "
          f"overall={report['synthetic'].get('overall_acc')}  "
          f"cmd={report['synthetic'].get('command_acc')}  "
          f"rej={report['synthetic'].get('reject_acc')}")


def selftest():
    """Cross-check CtcFsa.decode against a pure-Python oracle on random
    log-probs (many shapes, near-ties, blank-dominated)."""
    rows = _load_variations()
    phrases = [" ".join(ws) for _, _, _, _, ws in rows]
    dictionary = build_vocab(DEFAULT_DATA)
    words = sorted(dictionary)
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    fsa = CtcFsa(phrases, word2idx)
    rng = np.random.default_rng(0)
    bad = 0
    V = len(words) + 1   # +1 blank
    for trial in range(120):
        T = int(rng.integers(1, 120))
        kind = trial % 3
        if kind == 0:
            lp = -np.log(rng.random((T, V)) + 1e-6)
        elif kind == 1:                       # near-ties: quantized
            lp = -np.round(np.log(rng.random((T, V)) + 1e-6), 3)
        else:                                 # blank-dominated
            lp = -np.log(rng.random((T, V)) + 1e-6)
            lp[:, 0] = lp[:, 0] + 4.0
        a = fsa.decode(lp)
        b = _oracle_decode(fsa, lp)
        if (a[0] != b[0]) or (a[1] != b[1]):
            bad += 1
            if bad <= 3:
                print(f"MISMATCH trial={trial} T={T} kind={kind} "
                      f"decode=({a[0]:.4f},{a[1]}) oracle=({b[0]:.4f},{b[1]})")
    print(f"selftest: {120 - bad}/120 decode==oracle")
    return bad == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", default="test",
                    choices=["test", "holdout", "train"])
    ap.add_argument("--model", default=os.path.join(MODELS, "best.pt"))
    ap.add_argument("--reject-margin", type=float, default=8.0,
                    help="reject if (free_lp - constrained_lp)/frames > margin")
    ap.add_argument("--reject-content", action="store_true",
                    help="Expanded-vocab reject rule: reject when the free "
                         "decode does not look like a command (mostly "
                         "non-command words, or the FSA locked onto a 1-word "
                         "command for a multi-word utterance). Use with the "
                         "736-word vocab model.")
    ap.add_argument("--reject-empty", action="store_true",
                    help="ALSO reject when the free (greedy) decode is empty "
                         "(model emitted only blanks). Use with a model "
                         "trained on all-blank synthetic negatives.")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--report", default=None)
    # ---- sharding (multi-GPU) ----
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device", default=None,
                    help="cuda:0 / cuda:1 / cpu (default: cuda:0 if avail)")
    ap.add_argument("--merge", action="store_true",
                    help="merge the _part{i}.json shards into --report and "
                         "exit (no model needed)")
    ap.add_argument("--selftest", action="store_true",
                    help="cross-check the FSA decode vs an oracle and exit")
    args = ap.parse_args()
    if args.report is None:
        args.report = os.path.join(_HERE, f"test_v8_{args.split}_report.json")

    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)

    if args.merge:
        parts = []
        for i in range(args.num_shards):
            p = args.report.replace(".json", f".part{i}.json")
            if not os.path.exists(p):
                raise SystemExit(f"missing shard file: {p}")
            parts.append(json.load(open(p))["results"])
        results = [r for part in parts for r in part]
        report = build_report(results, args.split, args)
        json.dump(report, open(args.report, "w"), indent=1)
        print_report(report)
        print(f"  report      : {args.report}  ({args.num_shards} shards)")
        return

    dev = (torch.device(args.device) if args.device
           else torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(args.model, map_location=dev)
    words = ck["words"]
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    model = V8Model(ck["n_words"], d_model=ck["d_model"],
                    layers=ck["layers"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    if args.num_shards == 1:
        print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M "
              f"params  vocab={ck['n_words']}  device={dev}", flush=True)

    rows = _load_variations()
    phrases = [" ".join(ws) for _, _, _, _, ws in rows]
    fsa = CtcFsa(phrases, word2idx)

    rows = load_test(args.data, args.split)
    if args.num_shards > 1:
        rows = rows[args.shard_id::args.num_shards]
        part = args.report.replace(".json", f".part{args.shard_id}.json")
    else:
        part = args.report
    if args.num_shards == 1:
        print(f"{args.split} clips: {len(rows)}", flush=True)
    else:
        print(f"[shard {args.shard_id}/{args.num_shards}] "
              f"{args.split} clips: {len(rows)}  device={dev}", flush=True)

    def _feats(paths):
        """[T,80] mel per path (truncated to 800 frames), thread-parallel."""
        def one(p):
            x = read_wav16(p)
            m = log_mel(x)
            if m.shape[0] > 800:
                m = m[:800]
            return m
        with ThreadPoolExecutor(max_workers=8) as ex:
            return list(ex.map(one, paths))

    results = []
    t0 = time.time()
    n_batches = math.ceil(len(rows) / args.batch)
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        mels = _feats([p for (p, _, _) in chunk])
        lens = [m.shape[0] for m in mels]
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
            gap = (f_lp - c_lp) / max(1, Tj)
            if gap > args.reject_margin:
                pred = "REJECT"
            if args.reject_empty and not f_words:
                pred = "REJECT"
            if args.reject_content:
                # Expanded-vocab reject rule (the user's idea): OOS speech now
                # DECODES to real words, so reject when those words do NOT form
                # a command. Two content signals, either one rejects:
                #   (a) base_frac < 0.6  -- the free decode is mostly non-command
                #       (out-of-vocabulary) words, i.e. it does not look like a
                #       command.  (Empty free decode -> base_frac 0 -> reject.)
                #   (b) free_len>=2 and phrase_len/free_len < 0.20 -- the FSA
                #       locked onto a 1-word command for a multi-word utterance
                #       (e.g. 'time' for 'what time is it'), i.e. the words do
                #       not actually form the matched command.
                fw = [w - 1 for w in f_words]
                pw = [w - 1 for w in c_words] if c_words else []
                base_frac = (sum(1 for w in fw if w < 109) / len(fw)) if fw else 0.0
                plen_ratio = (len(pw) / len(fw)) if fw else 0.0
                if (base_frac < 0.6) or (len(fw) >= 2 and plen_ratio < 0.20):
                    pred = "REJECT"
            results.append({
                "file": os.path.basename(p),
                "gold": "REJECT" if gold == "OUT_OF_SCOPE" else gold,
                "pred": pred,
                "phrase": phrase,
                "free": " ".join([words[w - 1] for w in f_words]),
                "gap": round(float(gap), 4),
                "c_lp": round(float(c_lp), 3),
                "f_lp": round(float(f_lp), 3),
                "c_score": round(float(c_lp) / max(1, Tj), 4),
                "f_score": round(float(f_lp) / max(1, Tj), 4),
                "n_frames": int(Tj),
                "correct": pred == ("REJECT" if gold == "OUT_OF_SCOPE"
                                    else gold),
                "is_syn": is_syn,
                "asr_ms": round(enc_ms / len(chunk) + dec_ms, 1),
            })
        done = min(i + args.batch, len(rows))
        if done % 256 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)}  ({time.time() - t0:.0f}s)",
                  flush=True)

    if args.num_shards > 1:
        json.dump({"split": args.split, "shard": args.shard_id,
                   "results": results}, open(part, "w"), indent=1)
        print(f"[shard {args.shard_id}/{args.num_shards}] done in "
              f"{time.time() - t0:.0f}s -> {part}", flush=True)
    else:
        report = build_report(results, args.split, args)
        json.dump(report, open(args.report, "w"), indent=1)
        print_report(report)
        print(f"  report      : {args.report}")


if __name__ == "__main__":
    main()
