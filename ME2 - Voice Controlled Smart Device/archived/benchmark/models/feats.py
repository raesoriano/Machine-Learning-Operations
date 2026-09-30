"""Feature extraction for the ONNX VCM — re-exports the shared recipe.

Kept as a thin wrapper so benchmark/models stays self-contained while the
actual implementation lives in vcm.features (single source of truth).
"""
from vcm.features import (  # noqa: F401
    N_MELS, FRAME, HOP, log_mel,
)
