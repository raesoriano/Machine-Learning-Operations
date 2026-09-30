#!/usr/bin/env python3
import os, sys, time
sys.path.insert(0, "/home/ron.andrei.soriano/sandbox/VCM-v2/archive/ctc_v8")
t0=time.time()
print("T0 import vcm2.normalize", flush=True)
from vcm2.normalize import normalize
print(f"  imported {time.time()-t0:.2f}", flush=True)
t0=time.time()
r = normalize("alarm 6 AM")
print(f"  normalize('alarm 6 AM') -> {r!r} {time.time()-t0:.3f}", flush=True)
t0=time.time()
r2 = normalize("alarm six am")
print(f"  normalize('alarm six am') -> {r2!r} {time.time()-t0:.3f}", flush=True)
print("NORM_DONE", flush=True)
