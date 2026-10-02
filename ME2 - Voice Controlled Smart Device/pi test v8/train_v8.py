#!/usr/bin/env python3
"""pi test v8 -- train the causal Conformer + CTC encoder on the v6 dataset.

The original ME2 template architecture (CTC/attention causal encoder),
trained on the DGX box (this machine: 8x A100). Data: the AI231 ME2 v6
dataset -- in-scope train clips + the numerals split (digit phones),
same clip selection as the v6 HMM/GMM run so the models are comparable.

Run (2 GPUs):
    torchrun --nproc_per_node=2 train_v8.py --gpus 0,1
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from model import V8Model                       # noqa: E402
from data import (V8Dataset, collate, build_vocab, load_split,  # noqa: E402
                  load_test)

DEFAULT_DATA = "/home/ron.andrei.soriano/sandbox/data/external/me2-v6/dataset"


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        return rank, local, True
    return 0, 0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--out", default=os.path.join(_HERE, "models"))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32, help="per-GPU batch size")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--max-numerals", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rank, local, dist_on = setup_dist()
    dev = torch.device(f"cuda:{local}")
    torch.manual_seed(args.seed)

    os.makedirs(args.out, exist_ok=True)
    if rank == 0:
        print(f"device: {torch.cuda.get_device_name(local)}", flush=True)

    # ---------------- vocab (identical to v6) ---------------- #
    dictionary = build_vocab(args.data)
    words = sorted(dictionary)
    word2idx = {w: i + 1 for i, w in enumerate(words)}   # 0 = CTC blank
    V = len(words)
    if rank == 0:
        print(f"vocab: {V} words (+1 CTC blank)", flush=True)
        with open(os.path.join(args.out, "words.txt"), "w") as f:
            f.write("\n".join(words) + "\n")

    items = load_split(args.data, dictionary, word2idx,
                       max_numerals=args.max_numerals)
    ds = V8Dataset(items)
    sampler = DistributedSampler(ds, shuffle=True) if dist_on else None
    dl = DataLoader(ds, batch_size=args.bs,
                    sampler=sampler,
                    shuffle=(sampler is None),
                    num_workers=args.workers, collate_fn=collate,
                    pin_memory=True, drop_last=True)

    model = V8Model(V, d_model=args.d_model, layers=args.layers).to(dev)
    if dist_on:
        model = DDP(model, device_ids=[local])
    n_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"params: {n_params / 1e6:.1f} M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=0.01)
    total_steps = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup,
                           0.5 * (1 + math.cos(math.pi * s / total_steps))))
    lossf = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16: no scaler

    # validation set (holdout: 196 clips)
    from data import read_wav16, log_mel, ctc_target, tokenize_spoken
    import csv
    val_rows = []
    mpath = os.path.join(args.data, "holdout", "manifest.csv")
    with open(mpath) as f:
        for r in csv.DictReader(f):
            tr = (r.get("transcript") or "").strip()
            if not tr:
                continue
            ws = tokenize_spoken(tr)
            if ws and all(w in word2idx for w in ws):
                val_rows.append((os.path.join(args.data, "holdout",
                                              r["file"]),
                                 ctc_target([word2idx[w] for w in ws])))
    val_ds = V8Dataset(val_rows)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=4, collate_fn=collate)

    def val_loss(m):
        m.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for mel, lab, lens, llens in val_dl:
                mel = mel.to(dev, non_blocking=True)
                lab = lab.to(dev, non_blocking=True)
                ll = llens.to(dev, non_blocking=True)
                out = m(mel)                       # [B,T',V+1] logprobs
                T = out.shape[1]
                ilens = torch.full((mel.shape[0],), T, dtype=torch.long,
                                   device=dev)
                loss = lossf(out.transpose(0, 1).contiguous(), lab,
                             ilens, ll)
                tot += loss.item() * mel.shape[0]
                n += mel.shape[0]
        return tot / max(1, n)

    best = float("inf")
    t0 = time.time()
    step = 0
    for ep in range(args.epochs):
        if dist_on:
            sampler.set_epoch(ep)
        model.train()
        run = 0.0
        for mel, lab, lens, llens in dl:
            mel = mel.to(dev, non_blocking=True)
            lab = lab.to(dev, non_blocking=True)
            llens = llens.to(dev, non_blocking=True)
            T = mel.shape[1]
            # encoder output length: (T-2)/4 (two stride-2 convs)
            ilens = ((lens - 2) // 4).clamp(min=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(mel)
                loss = lossf(out.transpose(0, 1).contiguous(), lab,
                             ilens, llens)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            run += loss.item()
            if rank == 0 and step % 100 == 0:
                el = time.time() - t0
                print(f"  ep{ep} step{step} loss={run / 100:.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} ({el:.0f}s)",
                      flush=True)
                run = 0.0
        if rank == 0:
            vl = val_loss(model.module if dist_on else model)
            el = time.time() - t0
            print(f"epoch {ep}: val_loss={vl:.4f}  ({el:.0f}s)", flush=True)
            if vl < best:
                best = vl
                m = model.module if dist_on else model
                torch.save({"state_dict": m.state_dict(),
                            "words": words,
                            "d_model": args.d_model,
                            "layers": args.layers,
                            "n_words": V},
                           os.path.join(args.out, "best.pt"))
                print(f"  saved best.pt (val_loss={vl:.4f})", flush=True)
    if rank == 0:
        print(f"DONE in {time.time() - t0:.0f}s  best_val_loss={best:.4f}",
              flush=True)
    if dist_on:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
