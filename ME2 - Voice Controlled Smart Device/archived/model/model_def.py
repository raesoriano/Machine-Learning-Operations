"""Tiny VCM acoustic model (PyTorch).

A compact encoder that maps log-mel frames to CTC logits over the *constrained*
command vocabulary (vcm.VOCAB). CTC is chosen over RNN-T for a class project
because it trains simply, exports cleanly to ONNX, and — with a constrained
decoder — is exactly a "pure VCM": the model can only emit words it was told
about, so the output is always a parseable command transcript.

Architecture (target <= 2-3M params, runs on RPi4/5 CPU):
    log-mel [T,40]
      -> Conv1d stem (40 -> C, k=5)
      -> N x GRU-TCN block (residual, dilated)
      -> Linear(C -> |VOCAB|)   (CTC logits)

The decoder (constrained beam over VOCAB) lives in model/decode.py and is
shared by training-time eval, ONNX export, and the RPi service.
"""
import torch
import torch.nn as nn

from vcm.vocab import VOCAB, WORD2ID

BLANK = 0                      # CTC blank id
VOCAB_SIZE = len(VOCAB) + 1    # + blank


class TCNBlock(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1, dropout=0.1):
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(ch, ch, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pad = pad

    def forward(self, x):
        # x: [B, C, T]
        y = self.net(x)[:, :, : x.size(2)]
        return x + y


class VCMEncoder(nn.Module):
    """log-mel [B,T,40] -> CTC logits [B,T,|VOCAB|+1]."""

    def __init__(self, n_mels=40, channels=128, blocks=4, vocab_size=VOCAB_SIZE,
                 dropout=0.1):
        super().__init__()
        self.stem = nn.Conv1d(n_mels, channels, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList(
            [TCNBlock(channels, kernel=3, dilation=2 ** i, dropout=dropout)
             for i in range(blocks)]
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, mels):
        # mels: [B, T, 40]
        x = mels.transpose(1, 2)                 # [B, 40, T]
        x = torch.relu(self.stem(x))
        for b in self.blocks:
            x = b(x)
        x = x.transpose(1, 2)                    # [B, T, C]
        x = self.drop(x)
        return self.head(x)                      # [B, T, V]

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def ctc_loss(logits, targets, input_lengths, target_lengths, blank=BLANK):
    """CTC loss. logits [B,T,V] (log-softmax applied), targets [B,L] word ids
    (1-based, no blanks), lengths are int tensors."""
    logp = torch.log_softmax(logits, dim=-1)
    logp = logp.permute(1, 0, 2)                 # [T, B, V]
    return nn.functional.ctc_loss(
        logp, targets, input_lengths, target_lengths,
        blank=blank, zero_infinity=True)


if __name__ == "__main__":
    m = VCMEncoder()
    print(f"params: {m.count_params():,}")
    x = torch.randn(2, 100, 40)
    out = m(x)
    print(f"log-mel {tuple(x.shape)} -> ctc logits {tuple(out.shape)} "
          f"(vocab {VOCAB_SIZE} incl. blank)")
