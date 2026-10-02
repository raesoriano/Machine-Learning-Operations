"""Adapter for the PocketSphinx ensemble recognizer (v3 and v7).

v3 and v7 are the SAME script (`vcm_pi_v3.py`, byte-identical) -- they differ
only in the acoustic-model weights under `am/` (v3 = the original 1.6 MB LDA
AM; v7 = the custom CD AM retrained on the v6 dataset). So one adapter covers
both; `build("v3")` vs `build("v7")` just points at the different folder.

The production `Ensemble.classify()` returns the FINE 31-class label; this
adapter rolls it up to the COARSE 19-command schema so the shared live loop
can drive it with the same response map as v6/v8.
"""
from __future__ import annotations

import importlib.util
import os
import sys

from shared import _ME2, COARSE_INTENT

# fine 31-class (PocketSphinx / v6) -> coarse 19-command schema.
# Anything not listed maps to itself (it is already a coarse label).
FINE_TO_COARSE = {
    "BRIGHTNESS_20": "BRIGHTNESS", "BRIGHTNESS_60": "BRIGHTNESS",
    "BRIGHTNESS_100": "BRIGHTNESS",
    "COLOR_RED": "COLOR", "COLOR_GREEN": "COLOR", "COLOR_BLUE": "COLOR",
    "TIMER_10s": "TIMER", "TIMER_30s": "TIMER", "TIMER_1m": "TIMER",
    "ALARM_6_00AM": "ALARM", "ALARM_8_00AM": "ALARM", "ALARM_9_00PM": "ALARM",
    "TEMPERATURE_18": "TEMPERATURE", "TEMPERATURE_22": "TEMPERATURE",
    "TEMPERATURE_26": "TEMPERATURE",
    "CREATE_REMINDER_DRINK_WATER": "CREATE_REMINDER",
    "CREATE_REMINDER_EXERCISE": "CREATE_REMINDER",
    "CREATE_REMINDER_STUDY": "CREATE_REMINDER",
}


def _coarse(fine: str) -> str:
    return FINE_TO_COARSE.get(fine, fine)


def _load_module(name: str, folder: str):
    """Import `folder/vcm_pi_v3.py` under a unique module name (avoids the
    v3/v7 name collision if both were ever loaded in one process)."""
    path = os.path.join(_ME2, folder, "vcm_pi_v3.py")
    spec = importlib.util.spec_from_file_location(f"_vcm_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    # the script does `sys.path.insert(0, _HERE)` and `import vcm2` itself;
    # register it under its own name so its relative `vcm2` import resolves.
    sys.modules[f"_vcm_{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


class PocketSphinxAdapter:
    def __init__(self, name: str):
        folder = "pi test v3" if name == "v3" else "pi test v7"
        print(f"loading {name} PocketSphinx ensemble "
              f"({folder}/vcm_pi_v3.py) ...", flush=True)
        self._mod = _load_module(name, folder)
        self._ens = self._mod.Ensemble()
        self.name = name

    def classify_warm(self, pcm: bytes, reps: int = 1):
        fine, intent, transcript, prob = self._ens.classify_warm(pcm, reps=reps)
        coarse = _coarse(fine)
        return coarse, COARSE_INTENT.get(coarse, "unknown"), transcript, prob


def build(name: str, **kw) -> PocketSphinxAdapter:
    return PocketSphinxAdapter(name)
