"""Ground truth for `additional_test_data`.

Convention (per the data owner):
    folder name -> the INTENT (one of the 19 dataset labels)
    file name   -> what was SAID (the transcript); a trailing 1/2/3 is a
                   take number, not part of the utterance.

We map (folder, spoken phrase) -> one of the 31 commands (or REJECT). For
slot commands the slot value is parsed from the spoken phrase:
    "alarm 6 AM"          -> ALARM_6_00AM
    "brightness 100 percent" -> BRIGHTNESS_100
    "change color to blue"   -> COLOR_BLUE
    "temperature 22 degrees" -> TEMPERATURE_22
    "timer 10 seconds"     -> TIMER_10s
    "create a reminder to study" -> CREATE_REMINDER_STUDY
"""
import os
import re

from .normalize import normalize

# folder -> base command (no slot) for the non-slot intents
_SIMPLE = {
    "CALL": "CALL",
    "LIGHT_OFF": "LIGHT_OFF",
    "LIGHT_ON": "LIGHT_ON",
    "LIST_REMINDERS": "LIST_REMINDERS",
    "MESSAGE": "MESSAGE",
    "NEXT": "NEXT",
    "PAUSE": "PAUSE",
    "PLAY_MUSIC": "PLAY_MUSIC",
    "STOP": "STOP",
    "TIME": "TIME",
    "VOLUME_DOWN": "VOLUME_DOWN",
    "VOLUME_UP": "VOLUME_UP",
    "WEATHER": "WEATHER",
}

_ALARM = {"6": "ALARM_6_00AM", "8": "ALARM_8_00AM", "9": "ALARM_9_00PM"}
_BRIGHT = {"20": "BRIGHTNESS_20", "60": "BRIGHTNESS_60", "100": "BRIGHTNESS_100"}
_TEMP = {"18": "TEMPERATURE_18", "22": "TEMPERATURE_22", "26": "TEMPERATURE_26"}
_TIMER = {"10": "TIMER_10s", "30": "TIMER_30s", "1": "TIMER_1m"}
_COLOR = {"red": "COLOR_RED", "green": "COLOR_GREEN", "blue": "COLOR_BLUE"}
_REMINDER = {"drink water": "CREATE_REMINDER_DRINK_WATER",
             "exercise": "CREATE_REMINDER_EXERCISE",
             "study": "CREATE_REMINDER_STUDY"}


def _first_number(text):
    m = re.search(r"\d+", text)
    return m.group(0) if m else None


def gold_command(folder, spoken_phrase):
    """(folder, spoken phrase) -> one of the 31 commands, or REJECT."""
    folder = folder.upper()
    t = normalize(spoken_phrase)
    raw = (spoken_phrase or "").lower()          # keep digits for slot values
    if folder in _SIMPLE:
        return _SIMPLE[folder]
    if folder == "ALARM":
        return _ALARM.get(_first_number(raw), "REJECT")
    if folder == "BRIGHTNESS":
        return _BRIGHT.get(_first_number(raw), "REJECT")
    if folder == "TEMPERATURE":
        return _TEMP.get(_first_number(raw), "REJECT")
    if folder == "TIMER":
        return _TIMER.get(_first_number(raw), "REJECT")
    if folder == "COLOR":
        for c, cmd in _COLOR.items():
            if re.search(r"\b" + c + r"\b", t):
                return cmd
        return "REJECT"
    if folder == "CREATE_REMINDER":
        for a, cmd in _REMINDER.items():
            if a in t:
                return cmd
        return "REJECT"
    return "REJECT"


def spoken_phrase_from_filename(fname):
    """'alarm 6 AM 2.wav' -> 'alarm 6 AM' (strip .wav and trailing take N)."""
    base = fname[:-4] if fname.lower().endswith(".wav") else fname
    return re.sub(r" \d+$", "", base)


def build_ground_truth(data_dir):
    """data_dir (additional_test_data) -> list of
       {path, folder, spoken, gold_command}."""
    rows = []
    for folder in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, folder)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if not f.lower().endswith(".wav"):
                continue
            spoken = spoken_phrase_from_filename(f)
            rows.append({
                "path": os.path.join(p, f),
                "folder": folder,
                "spoken": spoken,
                "gold": gold_command(folder, spoken),
            })
    return rows


if __name__ == "__main__":
    import sys
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "additional_test_data")
    d = os.path.normpath(d)
    rows = build_ground_truth(d)
    from collections import Counter
    c = Counter(r["gold"] for r in rows)
    print(f"total clips: {len(rows)}")
    for k in sorted(c):
        print(f"  {k:30s} {c[k]}")
    print("\nsample rows:")
    for r in rows[:12]:
        print(f"  {r['folder']:16s} {r['spoken']!r:34s} -> {r['gold']}")
