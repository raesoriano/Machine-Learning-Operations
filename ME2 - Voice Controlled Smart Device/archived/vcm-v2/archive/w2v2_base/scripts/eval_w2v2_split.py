#!/usr/bin/env python3
"""ASR-only regression check on the ME2 held-out test split (6,470 clips).

Same guardrail the archived CTC pipeline used (archive/ctc_v8/scripts/
eval_asr_split.py): did the new ASR regress on the original (already-seen)
speakers? WER is computed the same way -- both gold and hypothesis are run
through the ME2 training tokenizer (vcm.parser normalization + digit ->
number-word expansion + constrained-vocab projection) and compared as word
lists -- so the number is directly comparable to the CTC baseline
(me2_v6: 47.7% WER on this split).

Usage:
    python backbone/scripts/eval_w2v2_split.py --split test \
        --report backbone/reports/asr_regression_w2v2.json
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACK = os.path.dirname(_HERE)
_REPO = os.path.dirname(_BACK)
_SANDBOX = os.path.dirname(_REPO)
_ME2 = os.path.join(_SANDBOX, "Machine-Learning-Operations",
                    "ME2 - Voice Controlled Smart Device")
sys.path.insert(0, _BACK)
sys.path.insert(0, _ME2)

from vcm2b.asr_w2v2 import ASRWav2Vec2  # noqa: E402
from model.train import tokenize        # noqa: E402
from vcm.vocab import ID2WORD           # noqa: E402


def wer_words(ref, hyp):
    d = np.zeros((len(ref) + 1, len(hyp) + 1), dtype=np.int32)
    for i in range(len(ref) + 1):
        d[i, 0] = i
    for j in range(len(hyp) + 1):
        d[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1,
                          d[i - 1, j - 1] + cost)
    return d[len(ref), len(hyp)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--art", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    asr = ASRWav2Vec2(art_dir=args.art, device=args.device)
    jsonl = os.path.join(_ME2, "data", "features", f"{args.split}.jsonl")
    rows = [json.loads(l) for l in open(jsonl)]
    if args.max_rows:
        rows = rows[:args.max_rows]
    print(f"evaluating {len(rows)} {args.split} rows "
          f"(baseline CTC me2_v6: 47.7% WER)", flush=True)

    tot_edits = tot_ref = exact = blank = 0
    for k, r in enumerate(rows):
        audio = r["audio"]
        if not os.path.isabs(audio):
            audio = os.path.join(_ME2, audio)
        x, sr = sf.read(audio, dtype="float32")
        hyp_text = asr.transcribe(x, sr=sr)
        if not hyp_text.strip():
            blank += 1
        ref = [ID2WORD[i - 1] for i in tokenize(r["text"])]
        hyp = [ID2WORD[i - 1] for i in tokenize(hyp_text)]
        tot_edits += wer_words(ref, hyp)
        tot_ref += len(ref)
        if hyp == ref:
            exact += 1
        if (k + 1) % 500 == 0:
            print(f"  {k + 1}/{len(rows)}  running WER "
                  f"{100 * tot_edits / max(tot_ref, 1):.1f}%", flush=True)

    out = {
        "model": "vcm-v2 backbone (wav2vec2-base-960h fine-tuned)",
        "split": args.split,
        "n_rows": len(rows),
        "wer": round(tot_edits / max(tot_ref, 1), 4),
        "word_exact_match": round(exact / len(rows), 4),
        "blank_rate": round(blank / len(rows), 4),
        "baseline_ctc_me2_v6_wer": 0.4766,
    }
    print(json.dumps(out, indent=2))
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
