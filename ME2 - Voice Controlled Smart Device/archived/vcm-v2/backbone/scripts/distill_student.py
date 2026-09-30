#!/usr/bin/env python3
"""Distill the Whisper encoder into a ~2.7M-param student (2-phase, cached).

Phase 1 (once): run the FROZEN Whisper encoder (72M params, GPU, batched)
    over all ME2 train+val clips and cache
        mel[1,80,3000] fp16, mask[1,3000] int8, teacher_emb[512] fp32, command
    to backbone/data/teacher_cache.npz  (~25 GB, disk has 900+ TB).
Phase 2: train the small student (2.7M params) from the cache -- no teacher
    in the loop, so each epoch is a few minutes on one GPU.

    student:  log-mel[1,80,T] -> conv stem (4x stride) -> 4x transformer
              -> masked mean-pool -> 512-d embedding
    loss = MSE(student_emb, teacher_emb) + 0.25 * CE(student_emb, command)

Phase 3: linear head (512 -> 32) trained on TRAIN, evaluated on VAL.
Phase 4: ONNX export (student + head, fp32 + int8) into pi_test/.

Usage:
    CUDA_VISIBLE_DEVICES=7 python backbone/scripts/distill_student.py            # full
    CUDA_VISIBLE_DEVICES=7 python backbone/scripts/distill_student.py --cache-only
    CUDA_VISIBLE_DEVICES=7 python backbone/scripts/distill_student.py --train-only
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (WhisperFeatureExtractor,
                          WhisperForConditionalGeneration)

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_ME2 = os.path.dirname(_REPO)
sys.path.insert(0, _ME2)

TEACHER = os.path.join(_BACK, "artifacts", "whisper_base_ft", "best")
OUT = os.path.join(_BACK, "artifacts", "student")
CACHE = os.path.join(_BACK, "data", "teacher_cache.npz")
MANIFEST = os.path.join(_ME2, "data", "manifests",
                        "positive_negative_manifest.csv")
EMBED = 512
N_LAYERS = 4
D_MODEL = 256
N_HEAD = 8
D_FF = 512
MAX_FRAMES = 3000
MAX_AUDIO_S = 12.0
CE_WEIGHT = 0.25
REJECT = "REJECT"


# ---------------------------------------------------------------- student
class ConvStem(nn.Module):
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=(1, 2), padding=1), nn.GELU(),
            nn.Conv2d(128, d, 3, stride=(1, 2), padding=1),
        )

    def forward(self, x):            # x: [B,80,T]
        x = x.unsqueeze(1)
        x = self.net(x)              # [B,d,5,T//16]
        x = x.mean(dim=2)
        return x.transpose(1, 2)     # [B,T//16,d]


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvStem()
        self.pos = nn.Parameter(torch.zeros(1, -(-MAX_FRAMES // 16), D_MODEL))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(D_MODEL, N_HEAD, D_FF, dropout=0.1,
                                       batch_first=True, norm_first=True)
            for _ in range(N_LAYERS)])
        self.norm = nn.LayerNorm(D_MODEL)
        self.proj = nn.Linear(D_MODEL, EMBED)

    def forward(self, mel, mask):
        h = self.stem(mel)                       # [B,T',d]
        T = h.size(1)
        need = T * 16
        if mask.size(1) < need:
            mask = nn.functional.pad(mask, (0, need - mask.size(1)))
        pmask = mask[:, :need].view(mask.size(0), T, 16).any(dim=2)
        h = h + self.pos[:, :T]
        h = self.norm(h)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=~pmask)
        w = pmask.float().unsqueeze(-1)
        emb = (h * w).sum(1) / w.sum(1).clamp(min=1e-6)
        return self.proj(emb)


# ---------------------------------------------------------------- phase 1
def load_rows(split):
    import csv
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            if r["polarity"] != "positive" or r["split"] != split:
                continue
            audio = r["audio"]
            if not os.path.isabs(audio):
                audio = os.path.join(os.path.dirname(os.path.dirname(MANIFEST)),
                                     audio)
            if not r["transcript"].strip() or not os.path.isfile(audio):
                continue
            rows.append((audio, r["command"]))
    return rows


def build_cache():
    import soundfile as sf
    import torchaudio
    device = "cuda"
    fe = WhisperFeatureExtractor.from_pretrained(TEACHER)
    teacher = WhisperForConditionalGeneration.from_pretrained(TEACHER)
    teacher.model.encoder.eval()
    teacher.to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    rows = load_rows("train") + load_rows("val")
    print(f"caching teacher embeddings for {len(rows)} clips ...", flush=True)
    mel = np.zeros((len(rows), 80, MAX_FRAMES), dtype=np.float16)
    mask = np.zeros((len(rows), MAX_FRAMES), dtype=np.int8)
    temb = np.zeros((len(rows), EMBED), dtype=np.float32)
    cmds = [None] * len(rows)

    def feat(a, sr):
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != 16000:
            a = torchaudio.functional.resample(
                torch.from_numpy(a), sr, 16000).numpy().astype(np.float32)
        if len(a) > 16000 * MAX_AUDIO_S:
            a = a[: int(16000 * MAX_AUDIO_S)]
        f = fe(a, sampling_rate=16000, return_tensors="np",
               padding="max_length")["input_features"][0].astype(np.float32)
        nf = min(int(round(len(a) / 16000 * 50)), MAX_FRAMES)
        return f, nf

    t0 = time.time()
    with torch.no_grad():
        for i, (audio, cmd) in enumerate(rows):
            a, sr = sf.read(audio, dtype="float32")
            f, nf = feat(a, sr)
            mel[i] = f.astype(np.float16)
            mask[i, :nf] = 1
            mf = torch.from_numpy(f).unsqueeze(0).to(device)
            mm = torch.zeros(1, MAX_FRAMES, dtype=torch.long, device=device)
            mm[0, :nf] = 1
            out = teacher.model.encoder(input_features=mf, attention_mask=mm,
                                        output_last_hidden_state=True,
                                        return_dict=True)
            h = out.last_hidden_state            # [1,1500,512]
            T2 = h.size(1)
            m2 = mm[:, :T2 * 2].view(1, T2, 2).any(dim=2).float().unsqueeze(-1)
            temb[i] = ((h * m2).sum(1) / m2.sum(1).clamp(min=1e-6)) \
                .cpu().numpy()[0]
            cmds[i] = cmd
            if (i + 1) % 2000 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1}/{len(rows)}  ({rate:.1f} clips/s, "
                      f"ETA {(len(rows) - i - 1) / rate / 60:.0f} min)",
                      flush=True)
    np.savez_compressed(CACHE, mel=mel, mask=mask, temb=temb,
                        commands=np.array(cmds))
    print(f"cache -> {CACHE}  "
          f"({os.path.getsize(CACHE) / 1e9:.1f} GB, {time.time() - t0:.0f}s)",
          flush=True)


# ---------------------------------------------------------------- phase 2
class CacheDS(Dataset):
    def __init__(self, cache, idx):
        self.c = cache
        self.idx = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        c = self.c
        return (torch.from_numpy(c["mel"][j].astype(np.float32)),
                torch.from_numpy(c["mask"][j].astype(np.int64)),
                torch.from_numpy(c["temb"][j]),
                str(c["commands"][j]))


def train_student(args, classes, c2i):
    device = "cuda"
    c = np.load(CACHE)
    # cache was written train-then-val; split the index accordingly
    n_tr = len(load_rows("train"))
    tr_idx = list(range(n_tr))
    va_idx = list(range(n_tr, len(c["temb"])))
    tr = DataLoader(CacheDS(c, tr_idx), batch_size=args.bs, shuffle=True,
                    num_workers=4, pin_memory=True)
    va = DataLoader(CacheDS(c, va_idx), batch_size=args.bs, shuffle=False,
                    num_workers=2, pin_memory=True)
    print(f"train={len(tr_idx)}  val={len(va_idx)}", flush=True)

    student = Student().to(device)
    print(f"student params: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(tr))

    best_val, t0 = 1e9, time.time()
    for epoch in range(args.epochs):
        student.train()
        tot, t_e = 0.0, time.time()
        for step, (mel, msk, tgt, cmds) in enumerate(tr):
            emb = student(mel.to(device), msk.to(device))
            ci = torch.tensor([c2i[x] for x in cmds], device=device)
            loss = nn.functional.mse_loss(emb, tgt.to(device)) \
                + CE_WEIGHT * nn.functional.cross_entropy(emb, ci)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            if (step + 1) % 100 == 0:
                print(f"  e{epoch} s{step + 1}/{len(tr)} loss={tot / (step + 1):.4f} "
                      f"({(time.time() - t_e) / (step + 1):.2f}s/step)", flush=True)
        student.eval()
        v, n = 0.0, 0
        with torch.no_grad():
            for mel, msk, tgt, cmds in va:
                emb = student(mel.to(device), msk.to(device))
                v += nn.functional.mse_loss(emb, tgt.to(device),
                                            reduction="sum").item()
                n += mel.size(0)
        v /= n
        print(f"epoch {epoch}: val_mse={v:.5f} ({time.time() - t0:.0f}s total)",
              flush=True)
        if v < best_val:
            best_val = v
            torch.save(student.state_dict(), os.path.join(OUT, "student.pt"))
            print(f"  saved best student (val_mse={v:.5f})", flush=True)
    student.load_state_dict(torch.load(os.path.join(OUT, "student.pt"),
                                       map_location=device))
    student.eval()
    return student, c, n_tr, va


def train_head(student, c, n_tr, va_idx, classes, c2i, device):
    tr = DataLoader(CacheDS(c, list(range(n_tr))), batch_size=256,
                    shuffle=True, num_workers=4, pin_memory=True)
    va = DataLoader(CacheDS(c, va_idx), batch_size=256, shuffle=False,
                    num_workers=2, pin_memory=True)
    head = nn.Linear(EMBED, len(classes)).to(device)
    ho = torch.optim.AdamW(head.parameters(), lr=1e-3)
    hs = torch.optim.lr_scheduler.CosineAnnealingLR(ho, T_max=5 * len(tr))
    best_acc, best_head = 0.0, None
    for ep in range(5):
        head.train()
        for mel, msk, tgt, cmds in tr:
            with torch.no_grad():
                emb = student(mel.to(device), msk.to(device))
            ci = torch.tensor([c2i[x] for x in cmds], device=device)
            loss = nn.functional.cross_entropy(head(emb), ci)
            ho.zero_grad()
            loss.backward()
            ho.step()
        hs.step()
        head.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for mel, msk, tgt, cmds in va:
                emb = student(mel.to(device), msk.to(device))
                pred = [classes[i] for i in head(emb).argmax(1)]
                correct += sum(p == g for p, g in zip(pred, cmds))
                n += len(cmds)
        acc = correct / n
        print(f"  head e{ep}: val_acc={100 * acc:.2f}%", flush=True)
        if acc > best_acc:
            best_acc, best_head = acc, {k: v.cpu().clone()
                                        for k, v in head.state_dict().items()}
    head.load_state_dict(best_head)
    head.to(device)
    print(f"head best val acc: {100 * best_acc:.2f}%", flush=True)
    torch.save(best_head, os.path.join(OUT, "head.pt"))
    with open(os.path.join(OUT, "classes.json"), "w") as f:
        json.dump({"classes": classes, "val_acc": best_acc}, f, indent=2)
    return head


def export_onnx(student, head, classes):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    PI = os.path.join(_REPO, "pi_test")

    class StudentOnnx(nn.Module):
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, mel, mask):
            return self.s(mel, mask)

    class HeadOnnx(nn.Module):
        def __init__(self, h):
            super().__init__()
            self.h = h

        def forward(self, emb):
            return self.h(emb)

    s_path = os.path.join(PI, "student.onnx")
    torch.onnx.export(StudentOnnx(student),
                      (torch.randn(1, 80, MAX_FRAMES),
                       torch.ones(1, MAX_FRAMES, dtype=torch.long)),
                      s_path,
                      input_names=["input_features", "attention_mask"],
                      output_names=["embedding"], opset_version=14,
                      dynamo=False,
                      dynamic_axes={"attention_mask": {1: "T"}})
    s_q = os.path.join(PI, "student_int8.onnx")
    quantize_dynamic(s_path, s_q, weight_type=QuantType.QInt8)
    h_path = os.path.join(PI, "head.onnx")
    torch.onnx.export(HeadOnnx(head), torch.randn(1, EMBED), h_path,
                      input_names=["embedding"], output_names=["logits"],
                      opset_version=14, dynamo=False)
    h_q = os.path.join(PI, "head_int8.onnx")
    quantize_dynamic(h_path, h_q, weight_type=QuantType.QInt8)
    with open(os.path.join(PI, "classes.json"), "w") as f:
        json.dump({"classes": classes}, f, indent=2)
    for p in (s_path, s_q, h_path, h_q):
        print(f"  {os.path.basename(p):22s} {os.path.getsize(p) / 1e6:8.3f} MB",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cache-only", action="store_true")
    ap.add_argument("--train-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    device = "cuda"

    if not args.train_only:
        build_cache()
    if args.cache_only:
        return

    classes = sorted(set(r[1] for r in load_rows("train"))
                     | set(r[1] for r in load_rows("val")) | {REJECT})
    c2i = {c: i for i, c in enumerate(classes)}
    print(f"{len(classes)} classes (incl REJECT)", flush=True)

    student, c, n_tr, va = train_student(args, classes, c2i)
    head = train_head(student, c, n_tr, va, classes, c2i, device)
    export_onnx(student, head, classes)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
