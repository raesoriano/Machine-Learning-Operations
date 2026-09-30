#!/usr/bin/env python3
"""OVERFIT PROBE: can wav2vec2-base + CTC + this decode learn the ME2 task at
all? Train on a tiny 300-clip subset for many epochs. If train CTC drops to
<5 and transcriptions match targets -> recipe is sound (LR/data issue). If it
stays ~200 -> fundamental bug in loss/target/feature setup.

This is a diagnostic, not the real model. Fast (~10 min on 1 GPU).
"""
import json
import os
import sys
import time

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCTC

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_SANDBOX = os.path.dirname(_REPO)
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
sys.path.insert(0, _ME2)
sys.path.insert(0, _BACK)

from model.train import tokenize        # noqa: E402
from vcm.vocab import ID2WORD, VOCAB    # noqa: E402
from model.decode import decode_to_text # noqa: E402

BLANK = 0
VOCAB_SIZE = len(VOCAB) + 1  # words + blank
MAX_SAMPLES = 16000 * 10


def load_rows(n=300):
    rows = [json.loads(l) for l in open(os.path.join(_BACK, "data", "train.jsonl"))]
    rows = rows[:n]
    for r in rows:
        r["_tok"] = tokenize(r["text"])
    return rows


class DS(Dataset):
    def __init__(self, rows):
        self.rows = rows
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x, sr = sf.read(r["audio"], dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        if sr != 16000:
            import torchaudio
            x = torchaudio.functional.resample(torch.from_numpy(x), sr, 16000).numpy()
        return torch.from_numpy(x), torch.tensor(r["_tok"], dtype=torch.long)


def collate(b):
    wavs, toks = zip(*b)
    max_len = min(max(w.shape[0] for w in wavs), MAX_SAMPLES)
    wavs = torch.stack([torch.cat([w, torch.zeros(max_len - w.shape[0])])[:max_len]
                        for w in wavs])
    ilens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    max_t = max(len(t) for t in toks)
    targets = torch.zeros(len(toks), max_t, dtype=torch.long)
    tlens = torch.tensor([len(t) for t in toks], dtype=torch.long)
    for i, t in enumerate(toks):
        targets[i, :len(t)] = torch.tensor(t)
    return wavs, targets, ilens, tlens


def main():
    device = "cuda"
    rows = load_rows(300)
    print(f"overfit probe: {len(rows)} clips, vocab {VOCAB_SIZE}", flush=True)
    dl = DataLoader(rows and DS(rows), batch_size=16, shuffle=True,
                    collate_fn=collate, num_workers=2)

    model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    with torch.no_grad():
        nn.init.normal_(model.wav2vec2.masked_spec_embed, 0.0, 0.02)
    model.lm_head = nn.Linear(model.config.hidden_size, VOCAB_SIZE)
    nn.init.zeros_(model.lm_head.weight)
    nn.init.zeros_(model.lm_head.bias)
    model.to(device)
    model.gradient_checkpointing_enable()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for ep in range(1, 16):
        model.train()
        total, n, t0 = 0.0, 0, time.time()
        for wavs, targets, ilens, tlens in dl:
            wavs = wavs.to(device)
            targets = targets.to(device)
            ilens = ilens.to(device)
            tlens = tlens.to(device)
            ilens_f = model._get_feat_extract_output_lengths(ilens)
            logits = model(wavs).logits
            logp = torch.log_softmax(logits.float(), dim=-1).permute(2, 0, 1).contiguous()
            ilens_f = ilens_f.clamp(max=logp.size(0))
            loss = ctc(logp, targets, ilens_f, tlens)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            total += loss.item() * len(wavs)
            n += len(wavs)
        print(f"ep {ep:02d}  train_ctc {total / max(n, 1):.3f}  ({time.time() - t0:.0f}s)",
              flush=True)

    # final: transcribe 10 clips
    model.eval()
    print("\n--- transcribe 10 clips (ref vs hyp) ---", flush=True)
    for r in rows[:10]:
        x, sr = sf.read(r["audio"], dtype="float32")
        x = x.mean(axis=1) if x.ndim > 1 else x
        with torch.no_grad():
            logits = model(torch.from_numpy(x)[None].to(device)).logits[0].float()
        hyp = decode_to_text(logits)
        ref = " ".join(ID2WORD[i - 1] for i in r["_tok"])
        mark = "OK " if hyp == ref else "   "
        print(f"  {mark} ref={ref!r:28s} hyp={hyp!r}", flush=True)


if __name__ == "__main__":
    main()
