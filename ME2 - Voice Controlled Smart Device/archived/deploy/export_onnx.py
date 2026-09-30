"""Export the trained VCM to ONNX (float32 -> quantize.py -> int8).

The exported graph is the ENTIRE acoustic model:
    input  "mels" [1, T, 40] float32   (16 kHz, 25 ms / 10 ms, log1p)
    output "logits" [1, T, |VOCAB|+1]  (CTC logits, blank=0)

Decoding (greedy CTC collapse) is done in numpy by the runtime
(benchmark/models + deploy/rpi_service), keeping the ONNX graph tiny and
portable. Constrained decoding is inherent: the head only has |VOCAB|+1
outputs, so the model can only ever emit command words.

Usage:
    python -m deploy.export_onnx --checkpoint model/checkpoints/best.pt \
        --config model/configs/tiny.yaml --out model/checkpoints/vcm_v1.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from model.model_def import VCMEncoder  # noqa: E402
from vcm.features import log_mel  # noqa: E402


class ExportWrapper(torch.nn.Module):
    """Wraps the encoder so the ONNX input is [1,T,40] mels (not [B,T,40])."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, mels):
        return self.encoder(mels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="torch .pt (state_dict)")
    ap.add_argument("--config", default=str(_REPO / "model/configs/tiny.yaml"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    model = VCMEncoder(n_mels=cfg["n_mels"], channels=cfg["channels"],
                       blocks=cfg["blocks"], dropout=cfg.get("dropout", 0.1))
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    wrapper = ExportWrapper(model)

    dummy = torch.from_numpy(np.zeros((1, 100, 40), dtype=np.float32))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, out,
        input_names=["mels"], output_names=["logits"],
        dynamic_axes={"mels": {1: "T"}, "logits": {1: "T"}},
        opset_version=13,
    )
    size_mb = out.stat().st_size / 1e6
    print(f"exported {out} ({size_mb:.2f} MB float32)")
    print("next: python -m deploy.quantize --in", out,
          "--out", str(out).replace(".onnx", "_int8.onnx"))


if __name__ == "__main__":
    main()
