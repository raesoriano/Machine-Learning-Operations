"""VCM benchmark — swappable test sets and swappable models.

Two axes of swappability (see COLLECTIVE_TASKS.md, WP4):

1. TEST SET  -- ``--testset synthetic`` (default, works today, no audio needed)
                ``--testset frozen``    (the real frozen v1 set, JSONL manifest
                with audio paths; drop-in, same schema)
2. MODEL     -- ``--model mock``    (stand-in: gold text + optional corruption;
                validates the harness end-to-end)
                ``--model onnx``    (your exported ONNX VCM)
                ``--model vosk``    (baseline: Vosk small + this parser)

The harness (evaluate.py) never changes when either axis is swapped.
"""
from .metrics import Metrics  # noqa: F401
