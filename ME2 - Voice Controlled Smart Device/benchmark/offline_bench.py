#!/usr/bin/env python3
"""Offline adaptation of the vcm-benchmark for ME2 models v3/v6/v7/v8.

The upstream vcm-benchmark is a LIVE tool: the laptop plays "wake word + command"
audio, the Raspberry Pi listens over SSH, and the laptop parses what the Pi
printed and scores it. That link cannot run headless here.

Per the task, this recreates the benchmark OFFLINE so it outputs the SAME
metrics: we feed each model the identical 202-clip class holdout set
(93 variations x 2 + 16 out-of-scope), decode each clip with the model's own
recognizer (same code path the Pi uses), map the model's fine-class output onto
the benchmark's 19-intent / 93-command / slot schema, and score it with the
benchmark's UNMODIFIED scoring code (vcmbench.report.score + metrics + schema +
slots). The only metric that needs the live link -- false-wake rate (commands
played WITHOUT a wake word) -- is reported as N/A offline.

Usage:
    python offline_bench.py v3 v6 v7 v8 v8new
    python offline_bench.py v8new         # one model

Runners (all under the ME2 base dir = parent of this folder):
    v3    -- PocketSphinx ensemble (archived/pi test v3)
    v6    -- from-scratch HMM/GMM (archived/pi test v6)
    v7    -- PocketSphinx ensemble, AM retrained on the v6 dataset (archived/pi test v7)
    v8    -- Conformer+CTC, 109-word vocab (pi test v8-conformer-ctc/models_neg/best.pt)
    v8neg -- same model, --reject-empty rule
    v8new -- Conformer+CTC, 736-word vocab + content-based reject rule
             (pi test v8-conformer-ctc/models/best.pt) -- the current model
"""
from __future__ import annotations
import csv
import importlib.util
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

HERE = Path(__file__).resolve().parent
# ME2 base dir = the parent of this benchmark/ folder (portable across machines).
ME2 = HERE.parent
HOLDOUT = Path("/tmp/holdout.parquet")
sys.path.insert(0, str(HERE))

from vcmbench import report as R          # noqa: E402  (unmodified benchmark scoring)
from vcmbench import schema as S          # noqa: E402
from vcmbench.schema import OOS, NONE, REJECT, SLOTTED  # noqa: E402

# ---------------------------------------------------------------------------
# holdout set
# ---------------------------------------------------------------------------
def load_holdout() -> list[dict]:
    if not HOLDOUT.exists():
        import urllib.request
        url = ("https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands/"
               "resolve/main/data/holdout-00000-of-00001.parquet")
        HOLDOUT.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    df = pd.read_parquet(HOLDOUT)
    clips = []
    for i, r in enumerate(df.to_dict("records")):
        data, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if sr != 16000:
            from vcmbench.audio import resample
            data = resample(data, sr, 16000)
        oos = int(r.get("out_of_scope") or 0) == 1 or r.get("command") == "OUT_OF_SCOPE"
        clips.append({
            "idx": i,
            "file": str(r.get("file", "")),
            "audio": data,
            "is_synthetic": int(r.get("is_synthetic") or 0) == 1,
            "true_intent": OOS if oos else str(r["command"]),
            "true_variation": "" if oos else str(r.get("variation") or ""),
            "true_slot": "" if oos else str(r.get("slot_value") or ""),
            "transcript": str(r.get("transcript", "")),
        })
    return clips

# ---------------------------------------------------------------------------
# fine-class -> benchmark (intent, slot, variation) schema
# ---------------------------------------------------------------------------
def _build_fine_map():
    """Map each model's 31 fine classes (+ REJECT) to the benchmark schema.

    The fine classes are the joint intent+slot names the models emit
    (TIMER_10s, TEMPERATURE_22, COLOR_RED, CREATE_REMINDER_DRINK_WATER, ...).
    Slot values are the benchmark's canonical strings (from variations.csv).
    """
    var = {}   # (intent, slot) -> first phrase
    order = []
    with open(HERE / "vcmbench" / "variations.csv") as f:
        for row in csv.DictReader(f):
            key = (row["label"], row["value"])
            if key not in var:
                var[key] = row["phrase"]
                order.append(key)
    slot_of = {
        "TIMER_10s": "10 seconds", "TIMER_30s": "30 seconds", "TIMER_1m": "1 minute",
        "ALARM_6_00AM": "6:00 AM", "ALARM_8_00AM": "8:00 AM", "ALARM_9_00PM": "9:00 PM",
        "TEMPERATURE_18": "18 degrees", "TEMPERATURE_22": "22 degrees", "TEMPERATURE_26": "26 degrees",
        "BRIGHTNESS_20": "20 percent", "BRIGHTNESS_60": "60 percent", "BRIGHTNESS_100": "100 percent",
        "COLOR_RED": "Red", "COLOR_BLUE": "Blue", "COLOR_GREEN": "Green",
        "CREATE_REMINDER_DRINK_WATER": "Drink water", "CREATE_REMINDER_STUDY": "Study",
        "CREATE_REMINDER_EXERCISE": "Exercise",
    }
    intent_of = {
        "PLAY_MUSIC": "PLAY_MUSIC", "WEATHER": "WEATHER", "TIME": "TIME",
        "LIGHT_ON": "LIGHT_ON", "LIGHT_OFF": "LIGHT_OFF", "PAUSE": "PAUSE",
        "STOP": "STOP", "NEXT": "NEXT", "VOLUME_UP": "VOLUME_UP", "VOLUME_DOWN": "VOLUME_DOWN",
        "CALL": "CALL", "MESSAGE": "MESSAGE", "LIST_REMINDERS": "LIST_REMINDERS",
        "TIMER_10s": "TIMER", "TIMER_30s": "TIMER", "TIMER_1m": "TIMER",
        "ALARM_6_00AM": "ALARM", "ALARM_8_00AM": "ALARM", "ALARM_9_00PM": "ALARM",
        "ALARM": "ALARM",  # v6's bare alarm class ("wake me up at X") -> ALARM intent, no slot
        "TEMPERATURE_18": "TEMPERATURE", "TEMPERATURE_22": "TEMPERATURE", "TEMPERATURE_26": "TEMPERATURE",
        "BRIGHTNESS_20": "BRIGHTNESS", "BRIGHTNESS_60": "BRIGHTNESS", "BRIGHTNESS_100": "BRIGHTNESS",
        "COLOR_RED": "COLOR", "COLOR_BLUE": "COLOR", "COLOR_GREEN": "COLOR",
        "CREATE_REMINDER_DRINK_WATER": "CREATE_REMINDER", "CREATE_REMINDER_STUDY": "CREATE_REMINDER",
        "CREATE_REMINDER_EXERCISE": "CREATE_REMINDER",
    }
    fine = {}
    for cls in intent_of:
        intent = intent_of[cls]
        slot = slot_of.get(cls, "")
        phrase = var.get((intent, slot), "")
        fine[cls] = (intent, slot, phrase)
    # v8 (Conformer+CTC) emits COARSE classes (no slot value): BRIGHTNESS,
    # ALARM, TIMER, COLOR, TEMPERATURE, CREATE_REMINDER. Map them to their
    # intent with an empty slot -> intent-correct, slot-wrong (a command-level
    # miss, not a misfire). This is honest: v8 does not decode the slot value.
    for cls, intent in [("BRIGHTNESS", "BRIGHTNESS"), ("ALARM", "ALARM"),
                        ("TIMER", "TIMER"), ("COLOR", "COLOR"),
                        ("TEMPERATURE", "TEMPERATURE"),
                        ("CREATE_REMINDER", "CREATE_REMINDER")]:
        fine[cls] = (intent, "", var.get((intent, ""), ""))
    fine["REJECT"] = (NONE, "", "")
    return fine

FINE = _build_fine_map()

def map_fine(cls: str):
    """fine class -> (pred_intent, pred_slot, pred_variation)."""
    if cls in FINE:
        return FINE[cls]
    # unknown / out-of-grammar: treat as an OTHER intent (a misfire, not a reject)
    return (f"OTHER:{cls}", "", "")

def pcm16(audio: np.ndarray) -> bytes:
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()

# ---------------------------------------------------------------------------
# model adapters -- each returns (fine_class, infer_ms)
# ---------------------------------------------------------------------------
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def run_v3(clips):
    m = _load_module("bench_v3", str(ME2 / "archived" / "pi test v3" / "vcm_pi_v3.py"))
    ens = m.Ensemble()
    out = []
    for c in clips:
        pcm = pcm16(c["audio"])
        t0 = time.perf_counter()
        cmd, _intent, _tr, _p = ens.classify(pcm)
        ms = (time.perf_counter() - t0) * 1000.0
        out.append((cmd, ms))
    return out

def run_v7(clips):
    m = _load_module("bench_v7", str(ME2 / "archived" / "pi test v7" / "vcm_pi_v3.py"))
    ens = m.Ensemble()
    out = []
    for c in clips:
        pcm = pcm16(c["audio"])
        t0 = time.perf_counter()
        cmd, _intent, _tr, _p = ens.classify(pcm)
        ms = (time.perf_counter() - t0) * 1000.0
        out.append((cmd, ms))
    return out

def run_v6(clips):
    m = _load_module("bench_v6", str(ME2 / "archived" / "pi test v6" / "vcm_pi_v6.py"))
    rec = m.V6Model()
    out = []
    for c in clips:
        pcm = pcm16(c["audio"])
        t0 = time.perf_counter()
        cmd, _intent, _tr, _ph = rec.classify(pcm)
        ms = (time.perf_counter() - t0) * 1000.0
        out.append((cmd, ms))
    return out

def run_v8(clips, model_path=None, reject_empty=False, reject_margin=8.0,
           reject_content=False):
    import math
    import torch
    v8dir = ME2 / "pi test v8-conformer-ctc"
    sys.path.insert(0, str(v8dir))
    sys.path.insert(0, str(ME2 / "archived" / "pi test v6"))
    ev8 = _load_module("bench_ev8", str(v8dir / "eval_v8.py"))
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ck = torch.load(model_path or (v8dir / "models" / "best.pt"), map_location=dev)
    words = ck["words"]
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    model = ev8.V8Model(ck["n_words"], d_model=ck["d_model"], layers=ck["layers"]).to(dev)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    rows = ev8._load_variations()
    phrases = [" ".join(ws) for _, _, _, _, ws in rows]
    fsa = ev8.CtcFsa(phrases, word2idx)
    from data import log_mel
    out = []
    for c in clips:
        x = c["audio"]
        m = log_mel(x)
        if m.shape[0] > 800:
            m = m[:800]
        T = m.shape[0]
        mel = torch.full((1, T, 80), math.log(1e-5), device=dev)
        mel[0, :T, :] = m.to(dev)
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(mel)
        o = o.float().cpu().numpy()[0]
        Tj = (T - 2) // 4 + 1
        lp = o[:Tj, :]
        c_lp, c_words = fsa.decode(lp)
        f_lp, f_words = ev8.greedy_ctc(lp)
        ms = (time.perf_counter() - t0) * 1000.0
        if c_words is not None:
            phrase = " ".join([words[w - 1] for w in c_words])
            fine = ev8.phrase_to_class(phrase)
        else:
            phrase, fine = "", "REJECT"
        if (f_lp - c_lp) / max(1, Tj) > reject_margin:
            fine = "REJECT"
        if reject_empty and not f_words:
            fine = "REJECT"
        if reject_content:
            # Expanded-vocab rule (same as eval_v8.py --reject-content):
            # OOS speech decodes to real words, so reject when those words
            # do NOT form a command.
            fw = [w - 1 for w in f_words]
            pw = [w - 1 for w in c_words] if c_words else []
            base_frac = (sum(1 for w in fw if w < 109) / len(fw)) if fw else 0.0
            plen_ratio = (len(pw) / len(fw)) if fw else 0.0
            if (base_frac < 0.6) or (len(fw) >= 2 and plen_ratio < 0.20):
                fine = "REJECT"
        out.append((fine, ms))
    return out

RUNNERS = {
    "v3": lambda clips: run_v3(clips),
    "v6": lambda clips: run_v6(clips),
    "v7": lambda clips: run_v7(clips),
    "v8": lambda clips: run_v8(clips, model_path=str(ME2 / "pi test v8-conformer-ctc" / "models_neg" / "best.pt")),
    "v8neg": lambda clips: run_v8(clips, model_path=str(ME2 / "pi test v8-conformer-ctc" / "models_neg" / "best.pt"), reject_empty=True),
    "v8new": lambda clips: run_v8(clips, model_path=str(ME2 / "pi test v8-conformer-ctc" / "models" / "best.pt"), reject_content=True),
}

# ---------------------------------------------------------------------------
# scoring (benchmark code, unmodified)
# ---------------------------------------------------------------------------
def score_model(name, clips, preds):
    trials = []
    for c, (fine, ms) in zip(clips, preds):
        pi, ps, pv = map_fine(fine)
        trials.append({
            "kind": "wake",
            "true_intent": c["true_intent"],
            "true_variation": c["true_variation"],
            "true_slot": c["true_slot"],
            "pred_intent": pi,
            "pred_slot": ps,
            "pred_variation": pv,
            "transcript": c["transcript"],
            "is_synthetic": c["is_synthetic"],
            "n_command_events": 0 if pi in (NONE, OOS) else 1,
            "infer_ms": round(ms, 3),
            "audio_ms": round(len(c["audio"]) / 16000.0 * 1000.0, 1),
            "latency_s": round(ms / 1000.0, 4),
            "wake_logged": False,
        })
    t0 = time.time()
    res = R.score(trials, [], {}, None, t0, time.time(), {"model": name, "mode": "offline"})
    res["model"] = name
    res["offline"] = True
    res["note"] = ("Offline adaptation of vcm-benchmark: same 202-clip holdout set, "
                   "model's own recognizer, benchmark's unmodified scoring code. "
                   "false-wake rate is N/A offline (needs the live wake-word link).")
    return res, trials

def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        o = float(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o

def main():
    only = sys.argv[1:] or ["v3", "v6", "v7", "v8", "v8new"]
    clips = load_holdout()
    n_oos = sum(1 for c in clips if c["true_intent"] == OOS)
    print(f"holdout: {len(clips)} clips ({len(clips)-n_oos} in-scope, {n_oos} OOS), "
          f"{sum(c['is_synthetic'] for c in clips)} synthetic", flush=True)
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
        # per-trial csv
        with open(outdir / f"{name}_trials.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "file", "true_intent", "true_variation", "true_slot",
                        "pred_intent", "pred_slot", "pred_variation", "is_synthetic",
                        "infer_ms", "correct_intent", "correct_command"])
            for c, t in zip(clips, trials):
                w.writerow([c["idx"], c["file"], t["true_intent"], t["true_variation"],
                            t["true_slot"], t["pred_intent"], t["pred_slot"],
                            t["pred_variation"], int(t["is_synthetic"]), t["infer_ms"],
                            t.get("correct_intent"), t.get("correct_command")])
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
            "slot_exact": (round(ov["slot_exact_rate"], 4) if ov.get("slot_exact_rate") is not None else None),
            "latency_p50_ms": round((ov.get("latency_p50") or 0) * 1000, 1),
            "latency_p95_ms": round((ov.get("latency_p95") or 0) * 1000, 1),
            "real_intent_acc": round(res["breakdowns"]["real voice"]["accuracy"], 4),
            "synth_intent_acc": round(res["breakdowns"]["synthetic voice"]["accuracy"], 4),
        }
        print(f"  -> {p.name}  intent_acc={summary[name]['intent_acc']} "
              f"cmd_acc={summary[name]['cmd_acc']} "
              f"FA={summary[name]['false_accept']} FR={summary[name]['false_reject']} "
              f"misfire={summary[name]['misfire']}", flush=True)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
