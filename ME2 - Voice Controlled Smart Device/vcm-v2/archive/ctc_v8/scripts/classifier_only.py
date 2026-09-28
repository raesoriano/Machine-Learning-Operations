"""Isolate the two stages: feed the classifier GOLD transcripts (no ASR).

This tells us the classifier's true capability, separate from the ASR's
new-speaker failure. If classifier-on-gold is high but end-to-end is low,
the ASR is the bottleneck.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vcm2.classifier import load_classifier, predict
from vcm2.ground_truth import build_ground_truth
from vcm2.pipeline import wer

_INTENT_OF = {
    "PLAY_MUSIC": "play_music", "WEATHER": "ask_question", "TIME": "ask_question",
    "LIGHT_ON": "lights_switch", "LIGHT_OFF": "lights_switch",
    "BRIGHTNESS_20": "lights_adjust", "BRIGHTNESS_60": "lights_adjust",
    "BRIGHTNESS_100": "lights_adjust", "COLOR_RED": "lights_adjust",
    "COLOR_GREEN": "lights_adjust", "COLOR_BLUE": "lights_adjust",
    "TIMER_10s": "set_timer", "TIMER_30s": "set_timer", "TIMER_1m": "set_timer",
    "ALARM_6_00AM": "set_alarm", "ALARM_8_00AM": "set_alarm",
    "ALARM_9_00PM": "set_alarm", "TEMPERATURE_18": "set_temperature",
    "TEMPERATURE_22": "set_temperature", "TEMPERATURE_26": "set_temperature",
    "PAUSE": "media_control", "STOP": "media_control", "NEXT": "media_control",
    "VOLUME_UP": "media_control", "VOLUME_DOWN": "media_control",
    "LIST_REMINDERS": "reminders_lists",
    "CREATE_REMINDER_DRINK_WATER": "reminders_lists",
    "CREATE_REMINDER_EXERCISE": "reminders_lists",
    "CREATE_REMINDER_STUDY": "reminders_lists", "CALL": "call",
    "MESSAGE": "call", "REJECT": "unknown",
}

clf = load_classifier()
rows = build_ground_truth("/mnt/jfs_hpc/home/ron.andrei.soriano/sandbox/additional_test_data")

cmd_ok = intent_ok = 0
reject = 0
mism = []
for r in rows:
    c, p = predict(clf, r["spoken"])          # GOLD spoken phrase -> classifier
    gi = _INTENT_OF.get(r["gold"], "unknown")
    pi = _INTENT_OF.get(c, "unknown")
    if c == r["gold"]:
        cmd_ok += 1
    if pi == gi:
        intent_ok += 1
    if c == "REJECT":
        reject += 1
    if c != r["gold"]:
        mism.append((r["folder"], r["spoken"], r["gold"], c, round(p, 3)))

n = len(rows)
print("=" * 60)
print(f"  CLASSIFIER-ONLY (gold transcripts, no ASR)  n={n}")
print("=" * 60)
print(f"  command accuracy : {100*cmd_ok/n:5.1f}%")
print(f"  intent accuracy  : {100*intent_ok/n:5.1f}%")
print(f"  REJECT emitted   : {reject} ({100*reject/n:.1f}%)")
print(f"\n  mismatches ({len(mism)}):")
for f, sp, gold, pred, p in mism:
    print(f"    {f:16s} {sp!r:34s} gold={gold:28s} -> {pred:28s} ({p})")
