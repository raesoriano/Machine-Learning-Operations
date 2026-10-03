#!/usr/bin/env python3
"""Build the ME2 positive/negative manifest for the tiny VCM (Option A: CTC).

Outputs (in data/manifests/):
  positive_negative_manifest.csv  - every usable audio row: polarity, 10-intent
                                    label, 31-command label, canonical target
                                    phrase (CTC target), slots, split, source
  summary.json                    - counts by polarity/intent/command/split
  frozen_test_v1.jsonl            - frozen benchmark test set (speaker-disjoint)

Design
------
* POSITIVE rows: audio whose meaning is one of the 31 fixed commands.
  - OptionB: all 17,924 active wavs (31 classes, 100 speakers, clean+noisy).
  - External: every row in metadata/standalone/processed_metadata.csv with a
    canonical label that is one of the 31 commands (FSC + SLURP + GSCv2).
  Each positive carries a CANONICAL TARGET PHRASE: the short in-vocab phrase
  the CTC model is trained to emit for that (command, slot value). The parser
  maps target -> (intent, slots); the benchmark compares (intent, slots).
* NEGATIVE rows: audio that must be rejected (intent=unknown).
  - External rows mapped to UNKNOWN (in-domain but ambiguous) or EXCLUDE
    (out of domain) in the metadata.
  - GSCv2 _background_noise_ (non-speech).
  - OptionB FLAGGED wavs (QC-excluded) are NOT used as negatives; they are
    simply excluded (their content is known-command audio that failed QC).
* Splits: OptionB keeps its speaker-disjoint 80/10/10 speaker split.
  External keeps the source dataset's split (train / valid|devel / test).
  The frozen test set = OptionB test speakers + external test-split rows.

Run:  python3 data/scripts/build_manifest.py   (from the ME2 folder)
"""
import csv
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)                      # .../ME2/.../data
ME2 = os.path.dirname(DATA)                       # .../ME2 - Voice Controlled Smart Device
OPT = os.path.join(DATA, "optionb")
EXT = os.path.join(DATA, "external")
META = os.path.join(DATA, "metadata", "standalone")
OUT = os.path.join(DATA, "manifests")

# --------------------------------------------------------------------------- #
# 31 commands -> (10-intent, canonical target phrase)
# The target is the short in-vocab phrase the CTC model is trained to emit for
# that command; the parser maps target -> (intent, slots).
# --------------------------------------------------------------------------- #
COMMANDS = {
    # command: (intent, target template)
    "PLAY_MUSIC":                       ("play_music", "play music"),
    "WEATHER":                          ("ask_question", "whats the weather"),
    "TIME":                             ("ask_question", "what time is it"),
    "LIGHT_ON":                         ("lights_switch", "turn on the lights"),
    "LIGHT_OFF":                        ("lights_switch", "turn off the lights"),
    "BRIGHTNESS_20":                    ("lights_adjust", "dim the lights to twenty percent"),
    "BRIGHTNESS_60":                    ("lights_adjust", "dim the lights to sixty percent"),
    "BRIGHTNESS_100":                   ("lights_adjust", "dim the lights to one hundred percent"),
    "COLOR_RED":                        ("lights_adjust", "set the lights to red"),
    "COLOR_GREEN":                      ("lights_adjust", "set the lights to green"),
    "COLOR_BLUE":                       ("lights_adjust", "set the lights to blue"),
    "TIMER_10s":                        ("set_timer", "set a timer for ten seconds"),
    "TIMER_30s":                        ("set_timer", "set a timer for thirty seconds"),
    "TIMER_1m":                         ("set_timer", "set a timer for one minute"),
    "ALARM_6_00AM":                     ("set_alarm", "set an alarm for six am"),
    "ALARM_8_00AM":                     ("set_alarm", "set an alarm for eight am"),
    "ALARM_9_00PM":                     ("set_alarm", "set an alarm for nine pm"),
    "TEMPERATURE_18":                   ("set_temperature", "set the temperature to eighteen degrees"),
    "TEMPERATURE_22":                   ("set_temperature", "set the temperature to twenty two degrees"),
    "TEMPERATURE_26":                   ("set_temperature", "set the temperature to twenty six degrees"),
    "PAUSE":                            ("media_control", "pause"),
    "STOP":                             ("media_control", "stop"),
    "NEXT":                             ("media_control", "next"),
    "VOLUME_UP":                        ("media_control", "volume up"),
    "VOLUME_DOWN":                      ("media_control", "volume down"),
    "LIST_REMINDERS":                   ("reminders_lists", "what are my reminders"),
    "CREATE_REMINDER_DRINK_WATER":      ("reminders_lists", "remind me to drink water"),
    "CREATE_REMINDER_EXERCISE":         ("reminders_lists", "remind me to exercise"),
    "CREATE_REMINDER_STUDY":            ("reminders_lists", "remind me to study"),
    "CALL":                             ("call", "call"),
    "MESSAGE":                          ("call", "message"),
}
INTENTS = sorted({v[0] for v in COMMANDS.values()})
assert len(COMMANDS) == 31 and len(INTENTS) == 10

# metadata canonical labels (32-class ontology) -> 31-command label
EXT_TO_CMD = {
    "PLAY_MUSIC": "PLAY_MUSIC", "WEATHER": "WEATHER", "TIME": "TIME",
    "LIGHT_ON": "LIGHT_ON", "LIGHT_OFF": "LIGHT_OFF",
    "VOLUME_UP": "VOLUME_UP", "VOLUME_DOWN": "VOLUME_DOWN",
    "PAUSE": "PAUSE", "STOP": "STOP", "NEXT": "NEXT",
    "LIST_REMINDERS": "LIST_REMINDERS", "CALL": "CALL",
    "COLOR_RED": "COLOR_RED", "COLOR_GREEN": "COLOR_GREEN", "COLOR_BLUE": "COLOR_BLUE",
    "ALARM_6_AM": "ALARM_6_00AM", "ALARM_8_AM": "ALARM_8_00AM", "ALARM_9_PM": "ALARM_9_00PM",
}
# NOTE: metadata has no BRIGHTNESS_*/TEMPERATURE_*/CREATE_REMINDER_* positives
# (directional heat/brightness and non-canonical reminders were mapped UNKNOWN
#  by the metadata policy). Those 9 commands are OptionB-only for v1.

# OptionB class -> slot dict (for benchmark gold)
def optionb_slots(label):
    if label.startswith("BRIGHTNESS_"):
        return {"percent": int(label.split("_")[1])}
    if label.startswith("TEMPERATURE_"):
        return {"value": int(label.split("_")[1]), "unit": "C"}
    if label.startswith("TIMER_"):
        return {"duration": label.split("_")[1]}
    if label.startswith("ALARM_"):
        return {"time": label.split("_")[1]}
    if label.startswith("CREATE_REMINDER_"):
        return {"action": {"DRINK_WATER": "drink water", "EXERCISE": "exercise",
                           "STUDY": "study"}[label.split("_", 2)[2]]}
    return {}

def ext_slots(cmd, slot_type, slot_value):
    if cmd.startswith("BRIGHTNESS_"):
        return {"percent": int(slot_value.split()[0])}
    if cmd.startswith("ALARM_"):
        return {"time": slot_value}
    if cmd == "LIST_REMINDERS":
        return {"action": "list"}
    if cmd == "CALL":
        return {}
    if cmd in ("WEATHER", "TIME"):
        return {"query": cmd.lower()}
    if cmd == "LIGHT_ON":
        return {"state": "on"}
    if cmd == "LIGHT_OFF":
        return {"state": "off"}
    if cmd.startswith("COLOR_"):
        return {"color": cmd.split("_")[1].lower()}
    if cmd == "VOLUME_UP":
        return {"action": "volume_up"}
    if cmd == "VOLUME_DOWN":
        return {"action": "volume_down"}
    if cmd == "PAUSE":
        return {"action": "pause"}
    if cmd == "STOP":
        return {"action": "stop"}
    if cmd == "NEXT":
        return {"action": "next"}
    if cmd == "PLAY_MUSIC":
        return {}
    return {}

# --------------------------------------------------------------------------- #
# audio path resolution (metadata audio_file -> real file under data/)
# --------------------------------------------------------------------------- #
def resolve_external(source, audio_file):
    """metadata audio_file -> path relative to data/ (or None if missing)."""
    base = os.path.basename(audio_file)
    if source == "fluent_speech_commands":
        # audio/fluent_speech_commands/<uuid>.wav
        # real: external/fluent_speech_commands/dataset/wavs/speakers/<spk>/<uuid>.wav
        # find by uuid (index built once)
        return _FSC_INDEX.get(base)
    if source == "slurp":
        # audio/slurp/real/<name>.wav  ->  external/slurp/real/<name>.flac
        # audio/slurp/synth/<name>.wav -> external/slurp/synth/<name>.flac
        m = re.match(r"audio/slurp/(real|synth)/(.+)$", audio_file)
        if not m:
            return None
        sub, name = m.group(1), os.path.basename(audio_file)
        cand = os.path.join("external", "slurp", sub, name.replace(".wav", ".flac"))
        return cand if os.path.exists(os.path.join(DATA, cand)) else None
    if source == "google_speech_commands_v2":
        # audio/google_speech_commands_v2/<class>/<file>.wav
        m = re.match(r"audio/google_speech_commands_v2/(.+)/(.+)$", audio_file)
        if not m:
            return None
        cand = os.path.join("external", "google_speech_commands_v2", m.group(1), m.group(2))
        return cand if os.path.exists(os.path.join(DATA, cand)) else None
    return None

_FSC_INDEX = {}

def build_fsc_index():
    root = os.path.join(EXT, "fluent_speech_commands", "dataset", "wavs", "speakers")
    for spk in os.listdir(root):
        d = os.path.join(root, spk)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".wav"):
                _FSC_INDEX[f] = os.path.join(
                    "external", "fluent_speech_commands", "dataset", "wavs", "speakers", spk, f)

# --------------------------------------------------------------------------- #
def _ood_row(rid, rel_path, source, split):
    """Generic out-of-domain speech negative (LibriSpeech narrative)."""
    return {
        "id": rid,
        "audio": rel_path,
        "polarity": "negative",
        "intent": "unknown",
        "command": "OOD_SPEECH",
        "target": "",
        "transcript": "",
        "slots": "{}",
        "split": split,
        "source": source,
        "speaker": "",
        "variant": "real",
    }

# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    build_fsc_index()
    rows = []
    n_missing = Counter()

    # ---- 1) OptionB positives (31 classes, speaker-disjoint split) -------- #
    with open(os.path.join(OPT, "manifest.csv")) as f:
        for r in csv.DictReader(f):
            if r["source"] != "optionb":
                continue
            cmd = r["label"]
            intent, target = COMMANDS[cmd]
            rows.append({
                "id": f"ob_{r['path'].replace('/', '_')[:-4]}",
                "audio": os.path.join("optionb", r["path"]),
                "polarity": "positive",
                "intent": intent,
                "command": cmd,
                "target": target,
                "transcript": r["transcript"],
                "slots": json.dumps(optionb_slots(cmd)),
                "split": r["split"],
                "source": "optionb",
                "speaker": r["speaker"],
                "variant": r["variant_id"],
            })

    # ---- 2) External rows from the metadata mapping ------------------------ #
    with open(os.path.join(META, "processed_metadata.csv")) as f:
        for r in csv.DictReader(f):
            src = r["source"]
            canon = r["canonical_label"]
            path = resolve_external(src, r["audio_file"])
            if path is None:
                n_missing[src] += 1
                continue
            split = {"train": "train", "train_synthetic": "train",
                     "valid": "val", "devel": "val",
                     "test": "test", "testing": "test", "validation": "val"}.get(
                         r["original_split"], "train")
            if canon in EXT_TO_CMD:
                cmd = EXT_TO_CMD[canon]
                intent, target = COMMANDS[cmd]
                rows.append({
                    "id": f"ext_{src}_{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)[:-4]}",
                    "audio": path,
                    "polarity": "positive",
                    "intent": intent,
                    "command": cmd,
                    "target": target,
                    "transcript": r["transcript"],
                    "slots": json.dumps(ext_slots(cmd, r["slot_type"], r["slot_value"])),
                    "split": split,
                    "source": src,
                    "speaker": r["speaker_id"],
                    "variant": "real",
                })
            else:  # UNKNOWN or EXCLUDE -> negative (reject)
                rows.append({
                    "id": f"neg_{src}_{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)[:-4]}",
                    "audio": path,
                    "polarity": "negative",
                    "intent": "unknown",
                    "command": "UNKNOWN" if canon == "UNKNOWN" else "EXCLUDE",
                    "target": "",
                    "transcript": r["transcript"],
                    "slots": "{}",
                    "split": split,
                    "source": src,
                    "speaker": r["speaker_id"],
                    "variant": "real",
                })

    # ---- 3) GSCv2 background noise -> negative ----------------------------- #
    bg = os.path.join(EXT, "google_speech_commands_v2", "_background_noise_")
    if os.path.isdir(bg):
        for i, f in enumerate(sorted(os.listdir(bg))):
            if f.endswith(".wav"):
                rows.append({
                    "id": f"neg_gsc_bg_{i:05d}",
                    "audio": os.path.join("external", "google_speech_commands_v2", "_background_noise_", f),
                    "polarity": "negative",
                    "intent": "unknown",
                    "command": "BACKGROUND_NOISE",
                    "target": "",
                    "transcript": "",
                    "slots": "{}",
                    "split": "train",
                    "source": "gsc_background_noise",
                    "speaker": "",
                    "variant": "real",
                })

    # ---- 4) LibriSpeech -> OOD speech negatives ---------------------------- #
    # Real human narrative speech, clearly out of the command domain. Strong
    # rejector training signal. The 1,500 curated (val) rows keep their split;
    # the full test-clean set is held out for the frozen OOD-reject test.
    ls_val = os.path.join(EXT, "librispeech")
    if os.path.isdir(ls_val):
        for i, f in enumerate(sorted(x for x in os.listdir(ls_val) if x.endswith(".flac"))):
            rows.append(_ood_row(f"neg_ls_{i:05d}",
                                 os.path.join("external", "librispeech", f),
                                 "librispeech", "train"))
    ls_tc = os.path.join(EXT, "librispeech_test_clean", "LibriSpeech")
    if os.path.isdir(ls_tc):
        # The 1,500 curated val files are a subset of test-clean (verified by
        # md5). Hold out only the 1,120 non-overlapping files for the frozen
        # OOD-reject test so no test clip was seen in training.
        import hashlib
        def _md5(p):
            with open(p, "rb") as fh:
                return hashlib.md5(fh.read()).hexdigest()
        cur_hashes = {_md5(os.path.join(ls_val, f))
                      for f in os.listdir(ls_val) if f.endswith(".flac")} \
            if os.path.isdir(ls_val) else set()
        n = 0
        for root, _dirs, files in os.walk(ls_tc):
            for f in sorted(files):
                if f.endswith(".flac"):
                    p = os.path.join(root, f)
                    if _md5(p) in cur_hashes:
                        continue
                    rows.append(_ood_row(f"neg_lstc_{n:05d}",
                                         os.path.relpath(p, DATA),
                                         "librispeech_test_clean", "test"))
                    n += 1

    # ---- write manifest ---------------------------------------------------- #
    fields = ["id", "audio", "polarity", "intent", "command", "target",
              "transcript", "slots", "split", "source", "speaker", "variant"]
    with open(os.path.join(OUT, "positive_negative_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- frozen test set --------------------------------------------------- #
    test_rows = [r for r in rows if r["split"] == "test"]
    with open(os.path.join(OUT, "frozen_test_v1.jsonl"), "w") as f:
        for r in test_rows:
            f.write(json.dumps({
                "id": r["id"],
                "audio": r["audio"],
                "text": r["target"] if r["polarity"] == "positive" else "",
                "intent": r["intent"],
                "slots": json.loads(r["slots"]),
                "subset": "in_domain" if r["polarity"] == "positive" else "ood_reject",
            }) + "\n")

    # ---- summary ----------------------------------------------------------- #
    pos = [r for r in rows if r["polarity"] == "positive"]
    neg = [r for r in rows if r["polarity"] == "negative"]
    summary = {
        "generated": "2026-09-27",
        "total_rows": len(rows),
        "positive": len(pos),
        "negative": len(neg),
        "intents": INTENTS,
        "commands": sorted(COMMANDS),
        "by_polarity_split": {p: dict(Counter(r["split"] for r in rows if r["polarity"] == p))
                              for p in ("positive", "negative")},
        "positive_by_command": dict(sorted(Counter(r["command"] for r in pos).items())),
        "positive_by_source": dict(Counter(r["source"] for r in pos)),
        "negative_by_command": dict(Counter(r["command"] for r in neg)),
        "negative_by_source": dict(Counter(r["source"] for r in neg)),
        "frozen_test": {"total": len(test_rows),
                        "in_domain": sum(1 for r in test_rows if r["polarity"] == "positive"),
                        "ood_reject": sum(1 for r in test_rows if r["polarity"] == "negative")},
        "missing_audio_skipped": dict(n_missing),
        "notes": [
            "Positive target = canonical in-vocab phrase; CTC trains to emit it, parser maps to (intent, slots).",
            "OptionB FLAGGED wavs excluded entirely (known-command audio that failed QC).",
            "BRIGHTNESS_*/TEMPERATURE_*/CREATE_REMINDER_* have OptionB audio only (metadata policy mapped external directional/other-value rows to UNKNOWN).",
            "Negatives = metadata UNKNOWN/EXCLUDE rows + GSCv2 _background_noise_ + LibriSpeech (1,500 val train; 1,120 non-overlapping test-clean held out as frozen OOD reject test).",
            "4,961 SLURP synth rows skipped: metadata references a different SLURP synth distribution than the 10,272 synth FLACs on disk (only 814 overlap); all 18,089 SLURP real rows resolved.",
            "Splits: OptionB speaker-disjoint 80/10/10; external keeps source splits.",
        ],
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
