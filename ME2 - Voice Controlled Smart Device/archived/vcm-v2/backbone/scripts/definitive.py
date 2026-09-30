#!/usr/bin/env python3
"""Definitive: whisper(GPU) + sklearn predict + wer_breakdown in ONE process,
mirroring the eval loop. Tests the ctranslate2/OpenMP double-runtime deadlock
and whether OMP_NUM_THREADS=1 fixes it."""
import os, sys, time
import soundfile as sf

_BACK = "/home/ron.andrei.soriano/sandbox/VCM-v2/backbone"
_REPO = "/home/ron.andrei.soriano/sandbox/VCM-v2"
sys.path.insert(0, _BACK)
sys.path.insert(0, os.path.join(_REPO, "archive", "ctc_v8"))
sys.path.insert(0, os.path.join(_BACK, "scripts"))

print("import asr_whisper", flush=True)
from vcm2b.asr_whisper import ASRWhisper
print("import classifier", flush=True)
from vcm2.classifier import load_classifier, predict
from vcm2.ground_truth import build_ground_truth
from vcm2.normalize import normalize
import importlib.util
spec = importlib.util.spec_from_file_location("ew", os.path.join(_BACK, "scripts", "eval_whisper.py"))
ew = importlib.util.module_from_spec(spec)
src = open(os.path.join(_BACK, "scripts", "eval_whisper.py")).read()
ns = {"__file__": os.path.join(_BACK, "scripts", "eval_whisper.py")}
exec(compile(src.split("def main")[0], "head", "exec"), ns)
wer_breakdown = ns["wer_breakdown"]

print("load asr (gpu)", flush=True)
asr = ASRWhisper(model_size="base.en", device="cuda", compute="float16")
print("load classifier", flush=True)
clf = load_classifier(os.path.join(_REPO, "archive", "ctc_v8", "artifacts", "classifier.pkl"))
print("build gt", flush=True)
rows = build_ground_truth("/home/ron.andrei.soriano/sandbox/additional_test_data")
print(f"n={len(rows)}", flush=True)

for k in range(3):
    r = rows[k]
    a, sr = sf.read(r["path"], dtype="float32")
    t0=time.time(); txt = asr.transcribe(a, sr=sr); ta=time.time()-t0
    t0=time.time(); cmd,prob = predict(clf, txt); tc=time.time()-t0
    t0=time.time(); w,s,ins,de = wer_breakdown(r["spoken"], txt); tw=time.time()-t0
    print(f"[{k}] asr={ta:.2f}s cls={tc:.3f}s wer={tw:.3f}s -> {cmd} (wer={w:.2f})", flush=True)
print("ALL_DONE", flush=True)
