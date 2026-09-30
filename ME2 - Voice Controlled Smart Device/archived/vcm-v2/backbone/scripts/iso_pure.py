#!/usr/bin/env python3
import time
import numpy as np
t0 = time.time()
def wer_breakdown(ref, hyp):
    ref = ref.split()
    hyp = hyp.split()[:200]
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    op = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(n + 1):
        d[i, 0] = i
        op[i, 0] = 2
    for j in range(m + 1):
        d[0, j] = j
        op[0, j] = 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j], op[i, j] = d[i - 1, j - 1], 0
                continue
            c_sub = d[i - 1, j - 1] + 1
            c_del = d[i - 1, j] + 1
            c_ins = d[i, j - 1] + 1
            best = min(c_sub, c_del, c_ins)
            d[i, j] = best
            op[i, j] = 1 if best == c_sub else (2 if best == c_del else 3)
    s = ins = de = 0
    i, j = n, m
    while i > 0 or j > 0:
        o = op[i, j]
        if o in (1, 2):
            i -= 1
        if o in (1, 3):
            j -= 1
        if o == 1:
            s += 1
        elif o == 2:
            de += 1
        else:
            ins += 1
    return d[n, m] / max(n, 1), s, ins, de
w, s, ins, de = wer_breakdown("alarm 6 AM", "alarm six am")
print(f"pure DP: {time.time()-t0:.4f}s wer={w:.3f}", flush=True)
print("PURE_DONE", flush=True)
