"""Re-export models/asr.onnx from the trained checkpoint (model on CPU)."""
import os, torch, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import models as M

ck = torch.load(os.path.join(HERE, "models", "asr.pt"), map_location="cpu")
model = M.ASRModel(n_mels=ck["n_mels"], n_frames=ck["n_frames"],
                   vocab=ck["vocab_size"],
                   feat_mean=ck.get("feat_mean", 0.0),
                   feat_std=ck.get("feat_std", 1.0))
model.load_state_dict(ck["state_dict"])
model = model.cpu().eval()
dummy = torch.randn(1, 40, ck["n_frames"], device="cpu")
out = os.path.join(HERE, "models", "asr.onnx")
torch.onnx.export(model, (dummy,), out,
                  input_names=["mel"], output_names=["logits"],
                  dynamic_axes={"mel": {0: "B"}, "logits": {0: "B"}},
                  opset_version=14)
print("exported", out, os.path.getsize(out), "bytes")

# sanity: onnxruntime round-trip
import onnxruntime as ort, numpy as np
s = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
x = np.random.randn(1, 40, ck["n_frames"]).astype(np.float32)
y = s.run(None, {"mel": x})[0]
print("onnx out", y.shape, "finite", bool(np.isfinite(y).all()))
