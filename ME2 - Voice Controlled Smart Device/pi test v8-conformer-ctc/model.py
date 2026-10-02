"""pi test v8 -- causal Conformer encoder + word-level CTC head.

The original ME2 template architecture: a causal (left-context only)
encoder that can stream frame-by-frame (no future context, no KV-cache
tricks needed for a 10 s command), with a CTC output over the word
vocabulary. Constrained decoding is a CTC-FSA dynamic program over the
93 command phrases (eval_v8.py); rejection is the score gap between the
constrained and the free (unigram) decode, same protocol as v6.

  * 16 kHz waveform -> log-mel front-end (torchaudio) -> conv front-end
  * 6-layer causal Conformer (d_model=256, nhead=4, d_ffn=1024)
  * CTC head: V words + 1 blank

Causality: every block uses only left context (causal self-attention with
a lower-triangular mask; convolutions are left-padded; FFN is pointwise).
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_causal_mask(T: int, device) -> torch.Tensor:
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


class ConvFrontend(nn.Module):
    """log-mel [B,T,80] -> [B,T/4,256] with stride-4 conv stack."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(80, d_model, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(d_model, d_model, 3, stride=1, padding=1), nn.GELU(),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, 80]
        x = x.transpose(1, 2)                    # [B, 80, T]
        x = self.net(x)                          # [B, d, T']
        x = x.transpose(1, 2)                    # [B, T', d]
        return self.norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.h = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B,H,T,h]
        att = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.h)
        att = att.masked_fill(~mask, float("-inf"))
        att = self.drop(att.softmax(dim=-1))
        y = torch.matmul(att, v)  # [B,H,T,h]
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.out(y)


class ConvBlock(nn.Module):
    """Depthwise causal conv (left-padded) with GLU."""

    def __init__(self, d_model: int, kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * d_model)
        self.depth = nn.Conv1d(d_model, d_model, kernel,
                               padding=kernel - 1, groups=d_model)
        self.norm2 = nn.GroupNorm(1, d_model)
        self.pw2 = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.norm(x)
        s = s.transpose(1, 2)                       # [B,C,T]
        s = self.pw1(s.transpose(1, 2)).transpose(1, 2)
        a, b = s.chunk(2, dim=1)
        s = a * torch.sigmoid(b)                    # GLU
        s = self.depth(s)[:, :, :x.shape[1]]        # causal trim
        s = self.norm2(s).transpose(1, 2)
        s = self.pw2(self.drop(s))
        return x + s


class ConformerBlock(nn.Module):
    """Causal conformer: x + 1/2*MH(x); MH = FFN/2 -> SA -> Conv -> FFN/2."""

    def __init__(self, d_model: int, nhead: int, d_ffn: int,
                 kernel: int = 15, dropout: float = 0.1):
        super().__init__()
        self.ff1 = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, d_ffn), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ffn, d_model), nn.Dropout(dropout))
        self.sa_norm = nn.LayerNorm(d_model)
        self.sa = CausalSelfAttention(d_model, nhead, dropout)
        self.conv = ConvBlock(d_model, kernel, dropout)
        self.ff2 = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, d_ffn), nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ffn, d_model), nn.Dropout(dropout))
        self.final = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ff1(x)
        x = x + self.sa(self.sa_norm(x), mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final(x)


class V8Model(nn.Module):
    """Causal Conformer + CTC head.

    forward(x_mel) -> log-probabilities [B, T', V+1] (log_softmax over CTC).
    """

    def __init__(self, n_words: int, d_model: int = 256, nhead: int = 4,
                 d_ffn: int = 1024, layers: int = 6, kernel: int = 15,
                 dropout: float = 0.1):
        super().__init__()
        self.frontend = ConvFrontend(d_model)
        self.pos = nn.Parameter(torch.zeros(1, 4096, d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            [ConformerBlock(d_model, nhead, d_ffn, kernel, dropout)
             for _ in range(layers)])
        self.ctc = nn.Linear(d_model, n_words + 1)
        self.d_model = d_model

    def forward(self, x_mel: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x_mel)
        T = x.shape[1]
        x = x + self.pos[:, :T, :]
        mask = make_causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return F.log_softmax(self.ctc(x), dim=-1)

    @torch.jit.export
    def encode(self, x_mel: torch.Tensor) -> torch.Tensor:
        """Return encoder states [B, T', d] (for the KV-cache streaming
        path / ONNX export)."""
        x = self.frontend(x_mel)
        T = x.shape[1]
        x = x + self.pos[:, :T, :]
        mask = make_causal_mask(T, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return x
