"""Frozen test set loader (the REAL benchmark, WP4).

Schema — one JSONL file, one row per utterance:

    {"id": "core_000123",
     "audio": "data/audio/test/core_000123.wav",   # relative to repo root
     "text": "set a timer for 5 minutes",          # gold transcript
     "intent": "set_timer",                        # gold intent
     "slots": {"duration": "5min"},                # gold slots (parser form)
     "subset": "core"}                             # core|oov|noise|farfield

The frozen v1 set is produced ONCE, versioned (benchmark/testset/frozen_v1/),
and never regenerated. To swap in v2, point --testset-manifest at the new
file — no code changes.

`load()` validates the schema and (optionally) that audio files exist, so a
broken manifest fails loudly before a 2-hour run.
"""
import json
from pathlib import Path

REQUIRED = ["id", "text", "intent", "slots", "subset"]
VALID_INTENTS = {
    "play_music", "ask_question", "lights_switch", "lights_adjust",
    "set_timer", "set_alarm", "set_temperature", "media_control",
    "reminders_lists", "call", "unknown",
}


def load(manifest, repo_root=None, check_audio=True):
    """Return (rows, warnings). rows: list of dicts (audio made absolute)."""
    manifest = Path(manifest)
    root = Path(repo_root) if repo_root else manifest.resolve().parents[2]
    rows, warnings = [], []
    seen = set()
    with manifest.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing = [k for k in REQUIRED if k not in row]
            if missing:
                warnings.append(f"line {i}: missing {missing}")
                continue
            if row["id"] in seen:
                warnings.append(f"line {i}: duplicate id {row['id']}")
            seen.add(row["id"])
            if row["intent"] not in VALID_INTENTS:
                warnings.append(f"line {i}: bad intent {row['intent']}")
            if "audio" in row:
                ap = Path(row["audio"])
                if not ap.is_absolute():
                    ap = root / ap
                row["audio"] = str(ap)
                if check_audio and not ap.exists():
                    warnings.append(f"line {i}: audio missing {ap}")
            rows.append(row)
    if not rows:
        raise ValueError(f"no valid rows in {manifest}")
    return rows, warnings
