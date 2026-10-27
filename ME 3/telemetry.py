"""
Shared telemetry for the ME 3 full-duplex voice agent.

The orchestrator WRITES latency samples and interaction records here; the
dashboard and the latency benchmark READ them. Two backends are kept in sync:

  1. An in-memory ring buffer (fast, same-process reads).
  2. A JSONL log file (append-only, so a separate process — e.g. the
     Streamlit dashboard — can read recent samples without sharing memory).

The JSONL file is the source of truth for cross-process consumers. Each line
is a single JSON object with a "type" field:

    {"type": "latency", "ts": 1730000000.0, "category": "asr_turnaround",
     "value_ms": 412.5, "turn": 7}
    {"type": "interaction", "ts": 1730000000.0, "role": "User",
     "text": "hello", "latency_ms": 412.5, "turn": 7}

The file path defaults to `telemetry.jsonl` next to this module, and can be
overridden with the TELEMETRY_FILE environment variable.
"""

import json
import os
import threading
import time
from collections import deque

# --- Configuration --------------------------------------------------------- #
_DEFAULT_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "telemetry.jsonl")
LOG_FILE = os.environ.get("TELEMETRY_FILE", _DEFAULT_LOG)

# Categories measured by the pipeline. Keep in sync with latency_benchmark.py.
CATEGORIES = (
    "vad_trigger",
    "asr_turnaround",
    "llm_ttfb",
    "tts_synthesis",
    "interruption",
)

# How many recent samples to keep in memory (per category).
_MAX_SAMPLES = 500
# How many recent interaction records to keep in memory.
_MAX_INTERACTIONS = 200


class Telemetry:
    """Thread-safe in-memory + JSONL telemetry sink/source."""

    def __init__(self, log_file: str = LOG_FILE):
        self.log_file = log_file
        self._lock = threading.Lock()
        # Per-category ring buffers of (ts, value_ms).
        self.latency: dict[str, deque] = {
            cat: deque(maxlen=_MAX_SAMPLES) for cat in CATEGORIES
        }
        # Interaction records, newest last.
        self.interactions: deque = deque(maxlen=_MAX_INTERACTIONS)
        # Monotonic turn counter (resets on process start).
        self._turn = 0

    # ------------------------------------------------------------------ #
    # Writers (used by the orchestrator)
    # ------------------------------------------------------------------ #
    def next_turn(self) -> int:
        """Increment and return the current turn counter."""
        with self._lock:
            self._turn += 1
            return self._turn

    def record_latency(self, category: str, value_s: float,
                       turn: int | None = None) -> float:
        """
        Record a latency sample. `value_s` is in SECONDS; it is stored and
        reported in milliseconds. Returns the value in ms.
        """
        value_ms = value_s * 1000.0
        ts = time.time()
        with self._lock:
            if category in self.latency:
                self.latency[category].append((ts, value_ms))
        self._append({
            "type": "latency",
            "ts": ts,
            "category": category,
            "value_ms": round(value_ms, 2),
            "turn": turn,
        })
        return value_ms

    def record_interaction(self, role: str, text: str,
                           latency_ms: float | None = None,
                           turn: int | None = None) -> None:
        """Record a single line of the interaction stream."""
        ts = time.time()
        rec = {
            "type": "interaction",
            "ts": ts,
            "role": role,
            "text": text,
            "latency_ms": (round(latency_ms, 2)
                           if latency_ms is not None else None),
            "turn": turn,
        }
        with self._lock:
            self.interactions.append(rec)
        self._append(rec)

    # ------------------------------------------------------------------ #
    # Internal JSONL append
    # ------------------------------------------------------------------ #
    def _append(self, obj: dict) -> None:
        try:
            line = json.dumps(obj, ensure_ascii=False)
            with self._lock:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
        except OSError as e:
            # Never let telemetry break the pipeline.
            print(f"[Telemetry] could not write log: {e}")

    # ------------------------------------------------------------------ #
    # Readers (used by dashboard / benchmark)
    # ------------------------------------------------------------------ #
    def samples(self, category: str) -> list[float]:
        """Return in-memory latency samples (ms) for a category, oldest first."""
        with self._lock:
            return [v for _, v in self.latency.get(category, ())]

    def has_data(self) -> bool:
        """True if any in-memory latency sample exists."""
        with self._lock:
            return any(self.latency[cat] for cat in CATEGORIES)

    def read_recent(self, max_lines: int = 500) -> list[dict]:
        """
        Read the most recent `max_lines` records from the JSONL log.
        This is the cross-process read path (e.g. the dashboard).
        """
        if not os.path.exists(self.log_file):
            return []
        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return []
        records = []
        for line in lines[-max_lines:]:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


# Module-level default instance (import and use directly).
default = Telemetry()
