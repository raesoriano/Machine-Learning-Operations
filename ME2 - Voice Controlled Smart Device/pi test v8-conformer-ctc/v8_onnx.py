#!/usr/bin/env python3
"""pi test v8-conformer-ctc -- standalone ONNX runtime (no torch, no torchaudio).

This is the Pi-side runtime for the v8 Conformer+CTC model. It needs only:

    pip install onnxruntime numpy

plus the Python stdlib. The log-mel front-end is baked into the ONNX graph
(export_onnx.py), so there is no torchaudio dependency anywhere. The 109-word
vocabulary and the 93 command phrases are baked into `meta.json` next to the
ONNX file, so there is no import from the `pi test v6` folder either.

Decoding is byte-for-byte the same protocol as eval_v8.py:
  * constrained decode : CTC-FSA dynamic program over the 93 command phrases
  * free decode        : greedy CTC (collapse repeats, drop blanks)
  * REJECT             : (free_lp - constrained_lp)/frames > margin, and/or
                         empty free decode (--reject-empty, for the
                         negatives-trained model)

Usage:
    python v8_onnx.py --model models/best.onnx --file clip.wav
    python v8_onnx.py --model models_neg/best.onnx --file clip.wav --reject-empty
    python v8_onnx.py --model models/best.onnx --data /path/to/dataset --split test
    python v8_onnx.py --model models_neg/best.onnx --data /path/to/dataset \
        --split test --reject-empty --report report.json
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import time
import wave

import numpy as np
import onnxruntime as ort

_HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# command ontology (copied verbatim from pi test v6/hgm/commands.py so this
# file is self-contained)
# ---------------------------------------------------------------------------

_RULES = [
    ("ALARM_6_00AM", lambda p: "alarm" in p and "six" in p),
    ("ALARM_8_00AM", lambda p: "alarm" in p and "eight" in p),
    ("ALARM_9_00PM", lambda p: "alarm" in p and "nine" in p),
    ("ALARM", lambda p: "alarm" in p or "wake me up" in p),
    ("BRIGHTNESS_100", lambda p: "brightness" in p and "hundred" in p),
    ("BRIGHTNESS_60", lambda p: "brightness" in p and "sixty" in p),
    ("BRIGHTNESS_20", lambda p: "brightness" in p and "twenty" in p),
    ("BRIGHTNESS", lambda p: "brightness" in p),
    ("COLOR_RED", lambda p: ("color" in p and "red" in p) or ("lights to red" in p) or ("set the lights to red" in p)),
    ("COLOR_GREEN", lambda p: ("color" in p and "green" in p) or ("lights to green" in p) or ("set the lights to green" in p)),
    ("COLOR_BLUE", lambda p: ("color" in p and "blue" in p) or ("lights to blue" in p) or ("set the lights to blue" in p)),
    ("COLOR", lambda p: "color" in p),
    ("TEMPERATURE_18", lambda p: "temperature" in p and "eighteen" in p),
    ("TEMPERATURE_22", lambda p: "temperature" in p and "twenty two" in p),
    ("TEMPERATURE_26", lambda p: "temperature" in p and "twenty six" in p),
    ("TEMPERATURE", lambda p: "temperature" in p),
    ("CREATE_REMINDER_DRINK_WATER", lambda p: "drink water" in p),
    ("CREATE_REMINDER_EXERCISE", lambda p: "exercise" in p),
    ("CREATE_REMINDER_STUDY", lambda p: "study" in p),
    ("CREATE_REMINDER", lambda p: ("create" in p and "reminder" in p) or "remind me" in p),
    ("LIST_REMINDERS", lambda p: "list my reminders" in p or "show my reminders" in p or p.strip() == "reminders"),
    ("TIMER_1m", lambda p: ("timer" in p or "countdown" in p) and "minute" in p),
    ("TIMER_30s", lambda p: ("timer" in p or "countdown" in p) and "thirty" in p),
    ("TIMER_10s", lambda p: ("timer" in p or "countdown" in p) and "ten" in p),
    ("TIMER", lambda p: "timer" in p or "countdown" in p),
    ("VOLUME_UP", lambda p: "volume up" in p or "increase the volume" in p or "turn the volume up" in p),
    ("VOLUME_DOWN", lambda p: "volume down" in p or "lower the volume" in p or "turn the volume down" in p),
    ("LIGHT_OFF", lambda p: "lights off" in p or "kill the lights" in p or "lights out" in p
                       or "shut off the lights" in p or "turn off the lights" in p),
    ("LIGHT_ON", lambda p: "lights on" in p or "turn on the lights" in p or "power on the lights" in p),
    ("PLAY_MUSIC", lambda p: "play music" in p or "play some music" in p or "start music" in p),
    ("NEXT", lambda p: "next song" in p or "play next song" in p or "skip song" in p),
    ("PAUSE", lambda p: "pause" in p),
    ("STOP", lambda p: "stop" in p or "end playback" in p),
    ("TIME", lambda p: "time" in p),
    ("WEATHER", lambda p: "weather" in p),
    ("CALL", lambda p: "call" in p),
    ("MESSAGE", lambda p: "message" in p),
]

CLASS_TO_INTENT = {
    "ALARM_6_00AM": "set_alarm", "ALARM_8_00AM": "set_alarm", "ALARM_9_00PM": "set_alarm",
    "ALARM": "set_alarm",
    "BRIGHTNESS_100": "lights_adjust", "BRIGHTNESS_60": "lights_adjust",
    "BRIGHTNESS_20": "lights_adjust", "BRIGHTNESS": "lights_adjust",
    "COLOR_RED": "lights_adjust", "COLOR_GREEN": "lights_adjust",
    "COLOR_BLUE": "lights_adjust", "COLOR": "lights_adjust",
    "TEMPERATURE_18": "set_temperature", "TEMPERATURE_22": "set_temperature",
    "TEMPERATURE_26": "set_temperature", "TEMPERATURE": "set_temperature",
    "CREATE_REMINDER_DRINK_WATER": "reminders_lists",
    "CREATE_REMINDER_EXERCISE": "reminders_lists",
    "CREATE_REMINDER_STUDY": "reminders_lists",
    "CREATE_REMINDER": "reminders_lists",
    "LIST_REMINDERS": "reminders_lists",
    "TIMER_1m": "set_timer", "TIMER_30s": "set_timer", "TIMER_10s": "set_timer",
    "TIMER": "set_timer",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIGHT_OFF": "lights_switch", "LIGHT_ON": "lights_switch",
    "PLAY_MUSIC": "play_music", "NEXT": "media_control", "PAUSE": "media_control",
    "STOP": "media_control", "TIME": "ask_question", "WEATHER": "ask_question",
    "CALL": "call", "MESSAGE": "call",
    "REJECT": "unknown",
}

_FINE_TO_COARSE = {
    "ALARM_6_00AM": "ALARM", "ALARM_8_00AM": "ALARM", "ALARM_9_00PM": "ALARM",
    "ALARM": "ALARM",
    "BRIGHTNESS_100": "BRIGHTNESS", "BRIGHTNESS_60": "BRIGHTNESS",
    "BRIGHTNESS_20": "BRIGHTNESS", "BRIGHTNESS": "BRIGHTNESS",
    "COLOR_RED": "COLOR", "COLOR_GREEN": "COLOR", "COLOR_BLUE": "COLOR",
    "COLOR": "COLOR",
    "TEMPERATURE_18": "TEMPERATURE", "TEMPERATURE_22": "TEMPERATURE",
    "TEMPERATURE_26": "TEMPERATURE", "TEMPERATURE": "TEMPERATURE",
    "CREATE_REMINDER_DRINK_WATER": "CREATE_REMINDER",
    "CREATE_REMINDER_EXERCISE": "CREATE_REMINDER",
    "CREATE_REMINDER_STUDY": "CREATE_REMINDER",
    "CREATE_REMINDER": "CREATE_REMINDER",
    "LIST_REMINDERS": "LIST_REMINDERS",
    "TIMER_1m": "TIMER", "TIMER_30s": "TIMER", "TIMER_10s": "TIMER",
    "TIMER": "TIMER",
    "VOLUME_UP": "VOLUME_UP", "VOLUME_DOWN": "VOLUME_DOWN",
    "LIGHT_OFF": "LIGHT_OFF", "LIGHT_ON": "LIGHT_ON",
    "PLAY_MUSIC": "PLAY_MUSIC", "NEXT": "NEXT", "PAUSE": "PAUSE",
    "STOP": "STOP", "TIME": "TIME", "WEATHER": "WEATHER",
    "CALL": "CALL", "MESSAGE": "MESSAGE",
    "REJECT": "REJECT",
}


def phrase_to_class(phrase: str) -> str:
    p = " ".join(phrase.lower().split())
    for cls, rule in _RULES:
        if rule(p):
            return cls
    return "REJECT"


def coarse_class(fine: str) -> str:
    return _FINE_TO_COARSE.get(fine, "REJECT")


# coarse 19-command schema -> intent (same map as eval_v8.py)
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


# ---------------------------------------------------------------------------
# decoding (copied verbatim from eval_v8.py)
# ---------------------------------------------------------------------------

class CtcFsa:
    """CTC-FSA decoder over the command grammar (same topology as v6)."""

    def __init__(self, phrases, word2idx):
        self.phrases = [p.split() if isinstance(p, str) else list(p)
                        for p in phrases]
        self.start = 0
        self.end = 1
        self.node_of = {}
        node = 2
        for p, ph in enumerate(self.phrases):
            for i in range(len(ph) + 1):
                self.node_of[(p, i)] = node
                node += 1
        self.n_nodes = node
        self.log_prior = -math.log(len(self.phrases))
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
        """lp: [T, V+1] log-posteriors (col 0 = blank).
        Returns (best_logprob, [word idx 1-based] or None)."""
        T = lp.shape[0]
        NEG = -1e30
        B = np.full(self.n_nodes, NEG)
        NB = np.full(self.n_nodes, NEG)
        B[0] = 0.0
        bp_B = np.zeros((T, self.n_nodes), dtype=np.int32)
        bp_NB = np.zeros((T, self.n_nodes), dtype=np.int32)
        bp_w = np.full((T, self.n_nodes), -1, dtype=np.int32)
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
            state[t] = (nNB > nB).astype(np.int8)
            B, NB = nB, nNB
        final = max(B[self.end], NB[self.end])
        if final <= NEG / 2:
            return NEG, None
        words = []
        v = self.end
        s = 1 if NB[self.end] > B[self.end] else 0
        for t in range(T - 1, -1, -1):
            if s == 1:
                u = int(bp_NB[t][v])
                w = int(bp_w[t][v])
                if w >= 0:
                    words.append(w + 1)
                v_prev = u
            else:
                v_prev = v
            if t == 0:
                break
            v = v_prev
            s = int(state[t - 1][v])
        words.reverse()
        return final, words


def greedy_ctc(lp: np.ndarray):
    """Greedy CTC decode. Returns (logprob, [word idx 1-based])."""
    pred = lp.argmax(axis=1)
    words, prev = [], -1
    for p in pred:
        if p != prev and p != 0:
            words.append(int(p))
        prev = int(p)
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


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------

def read_wav16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, f"expected 16 kHz, got {w.getframerate()}"
        assert w.getnchannels() == 1, "expected mono"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class V8Onnx:
    """The v8 Conformer+CTC recognizer backed by an ONNX session.

    decode(wav_path) -> dict with keys:
        phrase, fine, pred (coarse 19 schema or REJECT), free,
        c_lp, f_lp, reject (bool), asr_ms
    """

    def __init__(self, onnx_path: str, reject_margin: float = 8.0,
                 reject_empty: bool = False):
        self.words = None
        meta_path = os.path.join(os.path.dirname(os.path.abspath(onnx_path)),
                                 "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        self.words = meta["words"]
        self.word2idx = {w: i + 1 for i, w in enumerate(self.words)}
        self.fsa = CtcFsa(meta["phrases"], self.word2idx)
        self.reject_margin = reject_margin
        self.reject_empty = reject_empty
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, so, providers=["CPUExecutionProvider"])

    def _logprobs(self, x: np.ndarray) -> np.ndarray:
        """float32 mono waveform [T] -> [T', V+1] log-probs."""
        t = np.ascontiguousarray(x, dtype=np.float32).reshape(1, -1)
        out = self.sess.run(None, {"wav": t})[0]
        return out[0]

    def decode(self, wav_path: str) -> dict:
        t0 = time.perf_counter()
        x = read_wav16(wav_path)
        lp = self._logprobs(x)
        c_lp, c_words = self.fsa.decode(lp)
        f_lp, f_words = greedy_ctc(lp)
        asr_ms = (time.perf_counter() - t0) * 1000.0
        if c_words is not None:
            phrase = " ".join(self.words[w - 1] for w in c_words)
            fine = phrase_to_class(phrase)
            pred = coarse_class(fine)
        else:
            phrase, fine, pred = "", "REJECT", "REJECT"
        reject = ((f_lp - c_lp) / max(1, lp.shape[0]) > self.reject_margin)
        if self.reject_empty and not f_words:
            reject = True
        if reject:
            pred = "REJECT"
        return {
            "phrase": phrase, "fine": fine, "pred": pred,
            "free": " ".join(self.words[w - 1] for w in f_words),
            "c_lp": float(c_lp), "f_lp": float(f_lp),
            "reject": bool(reject), "asr_ms": round(asr_ms, 1),
        }


def load_test(data: str, split: str = "test"):
    rows = []
    mpath = os.path.join(data, split, "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            gold = (r.get("command") or "").strip()
            if not gold:
                continue
            p = os.path.join(data, split, r["file"])
            if os.path.exists(p):
                rows.append((p, gold, int(r.get("is_synthetic") or 0)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to the ONNX file (meta.json sits next to it)")
    ap.add_argument("--file", default=None, help="classify a single 16 kHz wav")
    ap.add_argument("--data", default=None,
                    help="dataset root (with <split>/manifest.csv) for eval mode")
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("--reject-margin", type=float, default=8.0)
    ap.add_argument("--reject-empty", action="store_true")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    rec = V8Onnx(args.model, reject_margin=args.reject_margin,
                 reject_empty=args.reject_empty)
    print(f"loaded {args.model}  vocab={len(rec.words)} "
          f"phrases={len(rec.fsa.phrases)}", flush=True)

    if args.file:
        r = rec.decode(args.file)
        print(json.dumps(r, indent=2))
        return

    if not args.data:
        ap.error("give --file <wav> or --data <dataset root>")
    rows = load_test(args.data, args.split)
    print(f"{args.split} clips: {len(rows)}", flush=True)
    results = []
    t0 = time.time()
    for i, (p, gold, is_syn) in enumerate(rows):
        r = rec.decode(p)
        results.append({
            "file": os.path.basename(p),
            "gold": "REJECT" if gold == "OUT_OF_SCOPE" else gold,
            "pred": r["pred"], "phrase": r["phrase"], "free": r["free"],
            "correct": r["pred"] == ("REJECT" if gold == "OUT_OF_SCOPE"
                                     else gold),
            "is_syn": is_syn, "asr_ms": r["asr_ms"],
        })
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"  {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)",
                  flush=True)

    n = len(results)
    inscope = [r for r in results if r["gold"] != "REJECT"]
    oos = [r for r in results if r["gold"] == "REJECT"]
    overall = sum(r["correct"] for r in results) / max(1, n)
    cmd_acc = sum(r["correct"] for r in inscope) / max(1, len(inscope))
    rej_acc = sum(r["correct"] for r in oos) / max(1, len(oos))
    intent_acc = sum(
        COARSE_TO_INTENT.get(r["pred"], "unknown") ==
        COARSE_TO_INTENT.get(r["gold"], "unknown") for r in results) / max(1, n)
    lat = np.array([r["asr_ms"] for r in results])
    per_class = {}
    for c in sorted(set(r["gold"] for r in results)):
        sub = [r for r in results if r["gold"] == c]
        per_class[c] = {"n": len(sub),
                        "acc": round(sum(r["correct"] for r in sub) /
                                     len(sub), 4)}

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
        "model": "v8 Conformer+CTC (ONNX standalone runtime)",
        "onnx": os.path.abspath(args.model),
        "split": args.split, "n_clips": n,
        "n_inscope": len(inscope), "n_oos": len(oos),
        "overall_acc": round(overall, 4),
        "command_acc": round(cmd_acc, 4),
        "reject_acc": round(rej_acc, 4),
        "intent_acc": round(intent_acc, 4),
        "real": _block([r for r in results if not r["is_syn"]]),
        "synthetic": _block([r for r in results if r["is_syn"]]),
        "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 1),
                       "p90": round(float(np.percentile(lat, 90)), 1)},
        "per_class": per_class,
        "clips": results,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "clips"},
                     indent=2))
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
