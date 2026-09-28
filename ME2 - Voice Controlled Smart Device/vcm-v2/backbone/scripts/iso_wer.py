#!/usr/bin/env python3
import os, sys, time
_HERE = "/home/ron.andrei.soriano/sandbox/VCM-v2/backbone/scripts"
_BACK = "/home/ron.andrei.soriano/sandbox/VCM-v2/backbone"
_REPO = "/home/ron.andrei.soriano/sandbox/VCM-v2"
sys.path.insert(0, _BACK)
sys.path.insert(0, os.path.join(_REPO, "archive", "ctc_v8"))
sys.path.insert(0, _HERE)
src = open(os.path.join(_HERE, "eval_whisper.py")).read()
ns = {"__file__": os.path.join(_HERE, "eval_whisper.py")}
exec(compile(src.split("def main")[0], "head", "exec"), ns)
wer_breakdown = ns["wer_breakdown"]
print("loaded", flush=True)
t0 = time.time()
for i in range(5):
    w, s, ins, de = wer_breakdown("alarm 6 AM", "alarm six am")
print(f"5x wer_breakdown done in {time.time()-t0:.3f}s -> {w:.3f}", flush=True)
print("ISO_DONE", flush=True)
