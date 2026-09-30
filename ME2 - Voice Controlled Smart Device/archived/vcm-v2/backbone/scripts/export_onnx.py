#!/usr/bin/env python3
"""Export the fine-tuned Whisper base.en (best) to ONNX for the Pi.

Two graphs:
  encoder.onnx : input_features[1,80,3000] (fp32) + attention_mask[1,3000] (int64)
                 -> last_hidden_state[1,3000,512]
  decoder.onnx : input_ids[1,T] (int64) + attention_mask[1,T] (int64)
                 + encoder_hidden_states[1,3000,512] (fp32)
                 -> logits[1,T,51864]

The decoder is used in a greedy loop from Python (prefix re-fed each step,
no KV cache -- fine for <=64 tokens). Mel extraction + tokenization + the
stage-2 classifier stay in Python (see pi_test/vcm_pi.py).

Usage:
    CUDA_VISIBLE_DEVICES=7 python backbone/scripts/export_onnx.py
"""
import os
import sys
import time

import numpy as np
import torch
from transformers import (AutoTokenizer, WhisperForConditionalGeneration,
                          WhisperFeatureExtractor)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
MODEL = os.path.join(_BACK, "artifacts", "whisper_base_ft", "best")
OUT = os.path.join(_REPO, "pi_test")
OPSET = 14


class EncoderMod(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_features, attention_mask):
        return self.enc(input_features=input_features,
                        attention_mask=attention_mask,
                        output_last_hidden_state=True,
                        return_dict=True).last_hidden_state


class DecoderMod(torch.nn.Module):
    def __init__(self, dec, lm_head):
        super().__init__()
        self.dec = dec
        self.lm_head = lm_head

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        out = self.dec(input_ids=input_ids,
                       attention_mask=attention_mask,
                       encoder_hidden_states=encoder_hidden_states,
                       return_dict=True)
        return self.lm_head(out.last_hidden_state)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    print(f"loading {MODEL} ...", flush=True)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL).to("cuda")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    TASK = getattr(model.generation_config, "task_token_id", None) \
        or model.config.bos_token_id
    print(f"loaded in {time.time() - t0:.1f}s  "
          f"(task_token_id={TASK}, eos={tokenizer.eos_token_id})", flush=True)

    # ---- reference (torch) greedy decode, for verification ----
    @torch.no_grad()
    def ref_transcribe(audio, sr):
        inputs = fe(audio, sampling_rate=sr, return_tensors="pt",
                    padding="max_length").to("cuda")
        n_frames = min(int(round(len(audio) / sr * 50)), 3000)
        mask = torch.zeros(1, 3000, dtype=torch.long)
        mask[0, :n_frames] = 1
        mask = mask.to("cuda")
        ids = model.generate(inputs.input_features, attention_mask=mask,
                             max_new_tokens=64, do_sample=False)
        if ids[0, 0] == tokenizer.pad_token_id:
            ids = ids[:, 1:]
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    # ---- export encoder ----
    print("exporting encoder ...", flush=True)
    enc_in = (torch.randn(1, 80, 3000, device="cuda"),
              torch.ones(1, 3000, dtype=torch.long, device="cuda"))
    torch.onnx.export(EncoderMod(model.model.encoder), enc_in,
                      os.path.join(OUT, "encoder.onnx"),
                      input_names=["input_features", "attention_mask"],
                      output_names=["last_hidden_state"],
                      opset_version=OPSET, dynamo=False,
                      dynamic_axes={"attention_mask": {1: "T"},
                                    "last_hidden_state": {1: "T"}})
    print(f"  encoder.onnx {os.path.getsize(os.path.join(OUT, 'encoder.onnx')) / 1e6:.1f} MB", flush=True)

    # ---- export decoder ----
    print("exporting decoder ...", flush=True)
    dec_in = (torch.tensor([[2577, 50257]], dtype=torch.long, device="cuda"),
              torch.ones(1, 2, dtype=torch.long, device="cuda"),
              torch.randn(1, 1500, 512, device="cuda"))
    torch.onnx.export(DecoderMod(model.model.decoder, model.proj_out), dec_in,
                      os.path.join(OUT, "decoder.onnx"),
                      input_names=["input_ids", "attention_mask",
                                   "encoder_hidden_states"],
                      output_names=["logits"],
                      opset_version=OPSET, dynamo=False,
                      dynamic_axes={"input_ids": {1: "T"},
                                    "attention_mask": {1: "T"},
                                    "encoder_hidden_states": {1: "S"},
                                    "logits": {1: "T"}})
    print(f"  decoder.onnx {os.path.getsize(os.path.join(OUT, 'decoder.onnx')) / 1e6:.1f} MB", flush=True)

    # ---- verify ONNX vs torch on 5 real clips ----
    import onnxruntime as ort
    import soundfile as sf
    print("verifying ONNX vs torch on 5 clips ...", flush=True)
    test_dir = os.path.join(_REPO, "test_data", "additional_test_data")
    clips = []
    for folder in sorted(os.listdir(test_dir)):
        p = os.path.join(test_dir, folder)
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".wav"):
                    clips.append(os.path.join(p, f))
        if len(clips) >= 5:
            break

    enc_s = ort.InferenceSession(os.path.join(OUT, "encoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    dec_s = ort.InferenceSession(os.path.join(OUT, "decoder.onnx"),
                                 providers=["CPUExecutionProvider"])
    EOS, PAD = tokenizer.eos_token_id, tokenizer.pad_token_id

    def onnx_transcribe(audio, sr):
        feat = fe(audio, sampling_rate=sr, return_tensors="np",
                  padding="max_length")["input_features"][0].astype(np.float32)
        n_frames = min(int(round(len(audio) / sr * 50)), 3000)
        mask = np.zeros((1, 3000), dtype=np.int64)
        mask[0, :n_frames] = 1
        hidden = enc_s.run(None, {"input_features": feat[None]})[0]
        ids = [TASK]
        for _ in range(64):
            din = np.array([ids], dtype=np.int64)
            dam = np.ones((1, len(ids)), dtype=np.int64)
            logits = dec_s.run(None, {"input_ids": din, "attention_mask": dam,
                                      "encoder_hidden_states": hidden})[0]
            nxt = int(logits[0, -1].argmax())
            ids.append(nxt)
            if nxt == EOS:
                break
        return tokenizer.decode(ids, skip_special_tokens=True).strip()

    ok = 0
    for c in clips:
        a, sr = sf.read(c, dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        r = ref_transcribe(a, sr)
        o = onnx_transcribe(a, sr)
        match = r == o
        ok += match
        print(f"  {'OK ' if match else 'DIFF'} {os.path.basename(c):34s} "
              f"torch={r[:40]!r} onnx={o[:40]!r}", flush=True)
    print(f"verify: {ok}/{len(clips)} exact matches", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
