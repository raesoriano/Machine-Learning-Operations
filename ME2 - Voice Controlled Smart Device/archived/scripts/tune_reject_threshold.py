#!/usr/bin/env python3
"""Tune the OOD-reject confidence threshold for the VCM.

The reject gate (model/decode.py: confidence / decode_with_confidence) decodes
an utterance to "" when its mean non-blank-frame log-prob is below a threshold,
so the parser returns {intent: "unknown"} and the device takes no action.

This script finds that threshold on the FROZEN v1 set, which is the only set
with labeled in-domain vs OOD rows (subset = in_domain | ood_reject):
    * in-domain reject rate  = false rejects (BAD — the device ignores a real
                               command). We want this low.
    * OOD reject rate        = true rejects  (GOOD — the device stays silent on
                               non-commands). We want this high.

It runs the EXACT shipping artifact (int8 ONNX, CPU) so the tuned number
transfers to the RPi, sweeps the threshold, and reports the operating point
that maximizes OOD rejection subject to an in-domain reject budget.

Usage:
    python scripts/tune_reject_threshold.py \
        --model model/checkpoints/me2_v6/vcm_int8.onnx \
        --manifest data/manifests/frozen_test_v1.jsonl \
        --data-root data \
        --id-reject-max 0.05 \
        --report reports/reject_threshold_v6.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.evaluate import _load_audio  # noqa: E402
from benchmark.models.feats import log_mel  # noqa: E402
from model.decode import confidence  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="int8 ONNX path")
    ap.add_argument("--manifest", required=True, help="frozen v1 JSONL")
    ap.add_argument("--data-root", default="data",
                    help="dir that manifest audio paths are relative to "
                         "(frozen v1 uses 'data')")
    ap.add_argument("--id-reject-max", type=float, default=0.05,
                    help="max allowed in-domain false-reject rate")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="debug: cap rows (per subset)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    import onnxruntime as ort
    session = ort.InferenceSession(args.model,
                                   providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    # (subset, confidence) per row
    id_conf, ood_conf = [], []
    t0 = time.time()
    n = 0
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            subset = row.get("subset", row.get("intent", ""))
            is_ood = (subset == "ood_reject") or (row.get("intent") == "unknown")
            if args.max_rows:
                if is_ood and len(ood_conf) >= args.max_rows:
                    continue
                if not is_ood and len(id_conf) >= args.max_rows:
                    continue
            apth = row["audio"]
            if not Path(apth).is_absolute():
                apth = str(Path(args.data_root) / apth)
            try:
                audio, sr = _load_audio(apth)
            except Exception as e:
                print(f"  skip {row['id']}: {e}")
                continue
            mel = log_mel(audio, sr=sr)
            logits = session.run(None, {in_name: mel[None]})[0][0]
            c = confidence(logits)
            (ood_conf if is_ood else id_conf).append(c)
            n += 1
    dt = time.time() - t0
    print(f"scored {n} rows ({len(id_conf)} in-domain, {len(ood_conf)} OOD) "
          f"in {dt:.0f}s")

    id_a = np.array(id_conf)
    ood_a = np.array(ood_conf)

    def report_row(thr):
        id_rej = float((id_a < thr).mean())
        ood_rej = float((ood_a < thr).mean())
        return id_rej, ood_rej

    def _finite(a):
        return a[np.isfinite(a)]

    id_f, ood_f = _finite(id_a), _finite(ood_a)
    id_blank = float((~np.isfinite(id_a)).mean())
    ood_blank = float((~np.isfinite(ood_a)).mean())

    # Sweep
    grid = np.round(np.arange(-6.0, 0.01, 0.05), 2)
    curve = []
    best = None
    for thr in grid:
        id_rej, ood_rej = report_row(float(thr))
        curve.append({"thr": float(thr), "id_reject": round(id_rej, 4),
                      "ood_reject": round(ood_rej, 4)})
        if id_rej <= args.id_reject_max:
            if best is None or ood_rej > best["ood_reject"]:
                best = {"thr": float(thr), "id_reject": round(id_rej, 4),
                        "ood_reject": round(ood_rej, 4)}

    print("\nconfidence stats (finite rows; all-blank rows decode to '' and "
          "are always rejected):")
    print(f"  in-domain : n={len(id_f)}  blank {id_blank*100:.1f}%  "
          f"mean {id_f.mean():.3f}  p10 {np.percentile(id_f,10):.3f}  "
          f"p50 {np.percentile(id_f,50):.3f}")
    print(f"  OOD       : n={len(ood_f)}  blank {ood_blank*100:.1f}%  "
          f"mean {ood_f.mean():.3f}  p50 {np.percentile(ood_f,50):.3f}  "
          f"p90 {np.percentile(ood_f,90):.3f}")
    print(f"\nbest operating point (id_reject <= {args.id_reject_max}):")
    if best:
        print(f"  thr = {best['thr']:.2f}   id_reject {best['id_reject']*100:.1f}%   "
              f"ood_reject {best['ood_reject']*100:.1f}%")
    else:
        print("  (no threshold meets the budget; loosen --id-reject-max)")

    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "model": args.model, "manifest": args.manifest,
            "id_reject_max": args.id_reject_max,
            "n_in_domain": len(id_conf), "n_ood": len(ood_conf),
            "id_blank_frac": round(id_blank, 4),
            "ood_blank_frac": round(ood_blank, 4),
            "best": best, "curve": curve,
        }, indent=2))
        print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
