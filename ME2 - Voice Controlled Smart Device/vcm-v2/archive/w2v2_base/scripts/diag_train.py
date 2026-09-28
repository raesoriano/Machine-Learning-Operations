#!/usr/bin/env python3
"""Quick diagnostic: data sample rates + target token lengths, and the
trained model's output length / per-sample CTC loss on a few val rows.
Used to confirm WHY the first fine-tune underfit (val CTC stuck ~243)."""
import json
import os
import sys

import numpy as np
import soundfile as sf

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_SANDBOX = os.path.dirname(_REPO)
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
sys.path.insert(0, _ME2)

from model.train import tokenize  # noqa: E402
from vcm.vocab import ID2WORD    # noqa: E402


def data_stats(jsonl, n=200):
    rows = [json.loads(l) for l in open(jsonl)]
    rows = rows[:n]
    lens, srs, durs = [], [], []
    for r in rows:
        a = r["audio"]
        if not os.path.isabs(a):
            a = os.path.join(_ME2, a)
        x, sr = sf.read(a, dtype="float32")
        srs.append(sr)
        durs.append(len(x) / sr)
        toks = tokenize(r["text"])
        lens.append(len(toks))
    print(f"[{os.path.basename(jsonl)}] n={len(rows)} "
          f"sr={sorted(set(srs))} dur={np.mean(durs):.2f}s "
          f"toklen mean={np.mean(lens):.2f} min={min(lens)} max={max(lens)}")
    # show a few examples
    for r in rows[:3]:
        toks = [ID2WORD[i - 1] for i in tokenize(r["text"])]
        print(f"    {r['text']!r:30s} -> {toks}")


def model_check(art, jsonl, n=8):
    import torch
    from transformers import AutoModelForCTC
    from safetensors.torch import load_file
    from model.decode import decode_to_text
    m = AutoModelForCTC.from_pretrained(art)
    vocab = json.load(open(os.path.join(art, "vocab.json")))
    m.lm_head = torch.nn.Linear(m.config.hidden_size, len(vocab["vocab"]) + 1)
    sd = load_file(os.path.join(art, "model.safetensors"))
    m.load_state_dict(sd, strict=False)
    m.eval()
    rows = [json.loads(l) for l in open(jsonl)][:n]
    print(f"[model {os.path.basename(art)}] output-vs-target on {n} val rows:")
    for r in rows:
        a = r["audio"]
        if not os.path.isabs(a):
            a = os.path.join(_ME2, a)
        x, sr = sf.read(a, dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(
                torch.from_numpy(x), sr, 16000).numpy()
        with torch.no_grad():
            logits = m(torch.from_numpy(x)[None]).logits[0].float()
        hyp = decode_to_text(logits)
        ref = [ID2WORD[i - 1] for i in tokenize(r["text"])]
        print(f"    ref={ref}  hyp={hyp.split()[:12]}")


if __name__ == "__main__":
    data_stats(os.path.join(_BACK, "data", "train.jsonl"))
    data_stats(os.path.join(_BACK, "data", "val.jsonl"))
    if "--model" in sys.argv:
        model_check(os.path.join(_BACK, "artifacts", "w2v2_base"),
                    os.path.join(_BACK, "data", "val.jsonl"))
