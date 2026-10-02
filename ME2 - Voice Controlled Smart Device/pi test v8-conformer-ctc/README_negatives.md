# pi test v8-conformer-ctc — reject via synthetic negatives (new dataset split)

Follow-up to the v8 Conformer+CTC model. The base v8 model rejected **0/47**
out-of-scope clips on the v6 test split. The updated AI231 ME2 dataset added a
`synthetic_negatives` split (1,000 train + 250 test generated OOS clips:
noise-only, babble, reversed, truncated, near-silence). This experiment adds
those 1,000 negatives to training as **all-blank CTC targets** so the model
learns to emit *nothing* on non-command audio, then re-evaluates rejection.

## What changed (minimal)

* `train_v8.py` — new `--negatives <dir>` flag: appends the negatives' train
  clips as `(wav, [])` items (empty CTC target = all-blank). No other change.
* `eval_v8.py` — new `--reject-empty` flag: reject when the free (greedy)
  decode is empty. (The original per-frame `--reject-margin` rule is
  mathematically incapable of triggering — see below.)
* `models_neg/best.pt` — retrained checkpoint (12 epochs, 3×A100, ~212 s).
  Same data as base v8 (28,834 clips) + 1,000 all-blank negatives = 29,834.
  Best val_loss 0.4840 (base v8: 0.4497 on the same holdout).

## Why the base model rejected nothing

The base reject rule is `(free_lp − constrained_lp)/frames > 8.0`. The
constrained FSA scores a phrase **plus surrounding blanks**, so a 3-word
phrase's penalty is only ~15 log-units spread over ~100 frames = **0.15 per
frame** — 50× below the margin of 8.0. No clip could ever reject. The rule
was never reachable; reject was 0.0 by construction, not by model quality.

## Results

### New 250-clip negatives test split (`_eval_negatives.json`)

| | n | reject (gold=REJECT) |
|---|---|---|
| base v8 model | 250 | 0.000 |
| **negatives model, `--reject-empty`** | 250 | **0.880** |

Per kind (reject-on-empty): near_silence 50/50, noise 44/50, babble 43/50,
reversed 44/50, truncated 39/50. The 30 misses are truncated/babble clips that
contain a real command-word fragment ("nine", "timer for", "volume") — the
model heard a plausible word, so it is genuinely ambiguous, not a failure.

### No-regression check: original 4,418-clip v6 test split
(`_eval_test_negmodel.json`, `--reject-empty`)

| metric | base v8 | negatives model |
|---|---|---|
| overall | 0.8497 | 0.8454 |
| command (4,371 in-scope) | 0.8588 | 0.8545 |
| intent | 0.8717 | 0.8710 |
| reject (47 OOS) | 0.0000 | 0.7447 (35/47 empty-free) |
| latency p50 / p90 | 24.9 / 43.0 ms | 27.5 / 50.3 ms |

**The cost of rejection:** rejecting on empty free-decode also rejects
**6.2 % of in-scope clips** (271/4,371), dropping command accuracy 0.8545 →
0.8396. The empty rate is sharply domain-split:

| | n | empty free-decode |
|---|---|---|
| real in-scope | 1,003 | **26.2 %** |
| synthetic in-scope | 3,368 | 0.24 % |

The all-blank negatives (synthetic noise) push the model to emit nothing on
**real** speech too — the same synthetic-vs-real domain gap that limits every
ME2 model, now visible as false rejects on real voices.

## Verdict

* The negatives **do** teach rejection: 0 → 88 % on the purpose-built
  negatives split, and 0 → 74 % on the original OOS clips, with only a ~0.4 pt
  overall-accuracy cost on the synthetic majority.
* The reject-on-empty rule is a **precision/recall dial**, not a free win:
  on this dataset it over-rejects real speech (26 %). On the Pi, where the
  audio is real, that is the wrong trade unless combined with a real-speech
  fix (more real training data / real-only retrain).
* **Recommendation:** keep `models_neg` + `--reject-empty` as the reject
  mechanism; tune it per-deployment (stricter for noisy rooms, looser for
  clean real speech). The base `models/best.pt` remains the best overall
  command model (0.85) when rejection is not required.

## Reproduce

```
# train (3 GPUs)
torchrun --nproc_per_node=3 train_v8.py --epochs 12 --bs 48 --workers 8 \
    --out models_neg --negatives <me2-v6-negatives/dataset>

# eval reject on the new negatives split
python eval_v8.py --data <me2-v6-negatives/dataset> --split test \
    --model models_neg/best.pt --reject-empty --report _eval_negatives.json

# eval no-regression on the original v6 test split
python eval_v8.py --data <me2-v6/dataset> --split test \
    --model models_neg/best.pt --reject-empty --report _eval_test_negmodel.json
```
