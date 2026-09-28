#!/usr/bin/env python3
"""Int8 dynamic-quantize the full Whisper ONNX graphs for the Pi.

encoder.onnx  -> pi_test/encoder_int8.onnx
decoder.onnx  -> pi_test/decoder_int8.onnx
"""
import os
from onnxruntime.quantization import QuantType, quantize_dynamic

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
PI = os.path.join(_REPO, "pi_test")

for name in ("encoder", "decoder"):
    src = os.path.join(PI, f"{name}.onnx")
    dst = os.path.join(PI, f"{name}_int8.onnx")
    print(f"quantizing {name} ...", flush=True)
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
    print(f"  {os.path.basename(src)} {os.path.getsize(src) / 1e6:8.1f} MB"
          f"  ->  {os.path.basename(dst)} {os.path.getsize(dst) / 1e6:8.1f} MB",
          flush=True)
print("done.", flush=True)
