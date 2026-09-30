"""Deterministic (intent, slots) -> one-of-31-command canonicalizer.

The VCM's job (per the ME2 task) is to understand the 31 fixed commands and
reject everything else. The benchmark therefore scores at the COMMAND level:
    audio -> model -> transcript -> parser.parse() -> (intent, slots)
         -> canonical.canonicalize() -> one of the 31 commands | "unknown"
and compares to the gold `command` column (in-domain) or the reject label
(OOD).

The 31 commands already encode the slot values (ALARM_6_00AM, BRIGHTNESS_20,
TEMPERATURE_22, ...), so command accuracy IS slot accuracy for the fixed set.
This is cleaner and more robust than exact slot-dict matching, and it matches
how a real device would act (it needs the command, not a slot dict).

Pure code, no model, no network. Same function used by the benchmark and the
RPi runtime.
"""

UNKNOWN = "unknown"

# The 31 fixed commands (see data/manifests/summary.json).
COMMANDS = {
    "PLAY_MUSIC",
    "WEATHER", "TIME",
    "LIGHT_ON", "LIGHT_OFF",
    "BRIGHTNESS_20", "BRIGHTNESS_60", "BRIGHTNESS_100",
    "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE",
    "TIMER_10s", "TIMER_30s", "TIMER_1m",
    "ALARM_6_00AM", "ALARM_8_00AM", "ALARM_9_00PM",
    "TEMPERATURE_18", "TEMPERATURE_22", "TEMPERATURE_26",
    "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
    "LIST_REMINDERS", "CREATE_REMINDER_DRINK_WATER",
    "CREATE_REMINDER_EXERCISE", "CREATE_REMINDER_STUDY",
    "CALL", "MESSAGE",
}


def _timer_cmd(duration):
    """'10s' -> TIMER_10s, '30s' -> TIMER_30s, '1min'/'1m' -> TIMER_1m."""
    if not duration:
        return None
    d = str(duration).lower()
    if d in ("10s", "10 sec", "10 seconds"):
        return "TIMER_10s"
    if d in ("30s", "30 sec", "30 seconds"):
        return "TIMER_30s"
    if d in ("1m", "1min", "1 minute", "1 min"):
        return "TIMER_1m"
    return None


def _alarm_cmd(time):
    """'6am' -> ALARM_6_00AM, '8am' -> ALARM_8_00AM, '9pm' -> ALARM_9_00PM."""
    if not time:
        return None
    t = str(time).lower().replace(":", "").replace(" ", "")
    if t in ("6am", "600am"):
        return "ALARM_6_00AM"
    if t in ("8am", "800am"):
        return "ALARM_8_00AM"
    if t in ("9pm", "900pm"):
        return "ALARM_9_00PM"
    return None


def canonicalize(intent, slots):
    """Map a parsed (intent, slots) to one of the 31 commands, or UNKNOWN.

    A correct intent with a slot value that is NOT one of the fixed commands
    (e.g. "7am" alarm, "5min" timer, "23 degrees") maps to UNKNOWN — for a
    fixed-command device that is a miss, not a command.
    """
    slots = slots or {}
    if intent == "play_music":
        return "PLAY_MUSIC"
    if intent == "ask_question":
        q = slots.get("query")
        if q == "weather":
            return "WEATHER"
        if q == "time":
            return "TIME"
        return UNKNOWN
    if intent == "lights_switch":
        if slots.get("state") == "on":
            return "LIGHT_ON"
        if slots.get("state") == "off":
            return "LIGHT_OFF"
        return UNKNOWN
    if intent == "lights_adjust":
        pct = slots.get("percent")
        if pct in (20, 60, 100):
            return f"BRIGHTNESS_{pct}"
        col = slots.get("color")
        if col in ("red", "green", "blue"):
            return f"COLOR_{col.upper()}"
        return UNKNOWN
    if intent == "set_timer":
        return _timer_cmd(slots.get("duration")) or UNKNOWN
    if intent == "set_alarm":
        return _alarm_cmd(slots.get("time")) or UNKNOWN
    if intent == "set_temperature":
        v = slots.get("value")
        if v in (18, 22, 26):
            return f"TEMPERATURE_{v}"
        return UNKNOWN
    if intent == "media_control":
        m = {"pause": "PAUSE", "stop": "STOP", "next": "NEXT",
             "volume_up": "VOLUME_UP", "volume_down": "VOLUME_DOWN"}
        return m.get(slots.get("action"), UNKNOWN)
    if intent == "reminders_lists":
        a = slots.get("action")
        if a == "list":
            return "LIST_REMINDERS"
        if a == "drink water":
            return "CREATE_REMINDER_DRINK_WATER"
        if a == "exercise":
            return "CREATE_REMINDER_EXERCISE"
        if a == "study":
            return "CREATE_REMINDER_STUDY"
        return UNKNOWN
    if intent == "call":
        if slots.get("action") == "message":
            return "MESSAGE"
        return "CALL"
    return UNKNOWN


if __name__ == "__main__":
    print(f"{len(COMMANDS)} commands")
    tests = [
        ("set_alarm", {"time": "6am"}), ("set_timer", {"duration": "10s"}),
        ("lights_adjust", {"percent": 100}), ("lights_adjust", {"color": "red"}),
        ("set_temperature", {"value": 22}), ("media_control", {"action": "next"}),
        ("call", {"action": "message"}), ("reminders_lists", {"action": "list"}),
        ("set_alarm", {"time": "7am"}), ("set_timer", {"duration": "5min"}),
    ]
    for i, s in tests:
        print(f"  {i:18s} {s} -> {canonicalize(i, s)}")
