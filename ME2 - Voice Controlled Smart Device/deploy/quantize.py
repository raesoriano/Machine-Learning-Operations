"""Dynamic-range int8 quantization of the VCM ONNX model.

Pure onnx + onnxruntime (no cloud, no GPU). Dynamic quantization needs no
calibration data and typically halves size with <1 point accuracy loss on a
command model — a good default for the RPi.

Usage:
    python -m deploy.quantize --in model/checkpoints/vcm_v1.onnx \
        --out model/checkpoints/vcm_v1_int8.onnx
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from onnxruntime.quantization import (  # local import: optional dep
        quantize_dynamic, QuantType)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(args.inp, str(out), weight_type=QuantType.QInt8)
    a, b = Path(args.inp).stat().st_size / 1e6, out.stat().st_size / 1e6
    print(f"quantized: {a:.2f} MB -> {b:.2f} MB ({b / a * 100:.0f}%)")
    if b > 10:
        print("WARNING: exceeds the 10 MB 'tiny' budget (PLAN.md)")


if __name__ == "__main__":
    main()
