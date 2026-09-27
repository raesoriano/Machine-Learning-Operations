"""Deterministic transcript -> command parser.

Pure code, no model, no network. This is the *same* parser used to:
  * score the benchmark (gold vs. predicted),
  * produce training labels, and
  * drive the RPi runtime service.

``parse(text)`` returns a ``Command(intent=..., slots={...})``. If nothing
matches, ``intent == "unknown"`` (a rejection), which the benchmark counts as
a false-accept/miss.
"""
import re
from dataclasses import dataclass, field

from . import slot_space as ss
from .numbers import words_to_int

UNKNOWN = "unknown"


@dataclass
class Command:
    intent: str
    slots: dict = field(default_factory=dict)

    def to_dict(self):
        return {"intent": self.intent, "slots": self.slots}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_NUM_WORDS = set()
for _w in ("zero", "one", "two", "three", "four", "five", "six", "seven",
           "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
           "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
           "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"):
    _NUM_WORDS.add(_w)
_NUM_WORDS |= {"hundred", "a", "an", "and"}


def _normalize(text: str) -> str:
    t = text.lower().strip()
    t = t.replace("’", "'")
    t = t.replace("what's", "whats")          # contractions before stripping
    t = re.sub(r"[^\w\s%]", " ", t)          # keep alnum, space, %
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _unit_alt(units):
    """Word-boundary alternation for unit words ('am' -> r'\bam\b')."""
    parts = []
    for u in units:
        if u.isalpha():
            parts.append(r"\b" + re.escape(u) + r"\b")
        else:
            parts.append(re.escape(u))
    return r"(?:" + "|".join(parts) + r")"


def _extract_int(text, units):
    """Find an integer (digits or number-words) immediately before a unit."""
    # 1) digits
    pat = r"(\d+)\s*" + _unit_alt(units)
    m = re.search(pat, text)
    if m:
        return int(m.group(1))
    # 2) number words: locate a unit, read back the trailing number words
    for m in re.finditer(_unit_alt(units), text):
        before = text[: m.start()].strip()
        if not before:
            continue
        toks = []
        for t in reversed(before.split()):
            if t in _NUM_WORDS:
                toks.append(t)
            else:
                break
        if toks:
            val = words_to_int(" ".join(reversed(toks)))
            if val is not None:
                return val
    return None


def _first_of(text, phrases):
    """Return the longest phrase (from a list) that appears in text, else None.

    Longest-first so 'best friend' wins over 'friend'.
    """
    for p in sorted(phrases, key=len, reverse=True):
        if re.search(r"\b" + re.escape(p) + r"\b", text):
            return p
    return None


# --------------------------------------------------------------------------- #
# per-intent rules
# --------------------------------------------------------------------------- #
_MUSIC_STOPWORDS = {"the", "a", "an", "and", "of", "to", "for", "my", "me",
                    "some", "with", "in", "on", "up"}


def _play_music(t):
    # A music command: a music noun (music/song/playlist/genre/artist) with a
    # music verb (play/start/resume/...), or a bare "resume"/"play"/"music".
    # Stopwords are excluded from the music-word set: artist names like
    # "The Weeknd" must not let "the" match "turn on the television".
    music_words = ((ss.phrases_to_words(ss.MUSIC_GENRES)
                    | ss.phrases_to_words(ss.MUSIC_ARTISTS)
                    | {"music", "song", "songs", "playlist", "track", "album",
                       "tune"})
                   - _MUSIC_STOPWORDS)
    has_music = any(re.search(r"\b" + re.escape(w) + r"\b", t)
                    for w in music_words)
    has_verb = re.search(
        r"\b(play|start|put|resume|continue|keep|find|listen|hear|bring|on)\b", t)
    bare = t in ("resume", "play", "music", "start")
    if not (bare or (has_music and has_verb)):
        return None
    # Strip the leading verb + articles so the query is what to play, not
    # "play jazz music" -> "jazz music".
    rest = re.sub(r"^(play|start|put|resume|continue|keep|find|listen|hear|"
                  r"bring)\s+", "", t).strip()
    rest = re.sub(r"^(some|the|a|an|my|me)\s+", "", rest).strip()
    query = rest or None
    if query in ("music", "some music", "the music"):
        query = None
    return Command("play_music", {"query": query} if query else {})


# Closed question types the device answers (VCM is not a general Q&A bot).
# (regex, canonical query type). Only weather/time map to the 31 commands;
# the rest are out-of-scope questions (canonicalize -> unknown).
_QUESTION_PATTERNS = [
    (r"\bweather\b", "weather"),
    (r"\bwhat time\b", "time"),
    (r"\btemperature outside\b", "weather"),
    (r"\b(rain|rainy|raining|forecast|snow|snowy|umbrella|sunny)\b", "weather"),
    (r"\bis it (hot|cold|warm)\b", "weather"),
    (r"\bwho won\b", "sports"),
    (r"\bhow far\b", "distance"),
    (r"\bwhat day\b", "date"),
]


def _ask_question(t):
    for pat, qtype in _QUESTION_PATTERNS:
        if re.search(pat, t):
            return Command("ask_question", {"query": qtype})
    # Bare "time" / "tell me the time" / "what is the time" -> time query.
    if re.search(r"\btime\b", t) and not re.search(r"\btimer\b", t):
        return Command("ask_question", {"query": "time"})
    return None


def _set_temperature(t):
    if re.search(r"\blight(s)?\b", t):
        return None  # "set the lights to cool" is a color, not a temp
    if re.search(r"\b(what|whats|how)\b", t):
        return None  # "what's the temperature outside" is a question
    if not re.search(r"\b(temperature|thermostat|heat|cool|ac)\b", t):
        return None
    val = _extract_int(t, ["degrees", "degree", "celsius", "fahrenheit", "°"])
    if val is None:
        val = _extract_int(t, ["c", "f"])
    # A bare "what's the temperature outside" is a question, not a set command.
    if val is None and not re.search(r"\bset\b", t):
        return None
    slots = {}
    if val is not None:
        slots["value"] = val
        if re.search(r"\bf\b|\bfahrenheit\b", t):
            slots["unit"] = "F"
        elif re.search(r"\bc\b|\bcelsius\b", t):
            slots["unit"] = "C"
        else:
            slots["unit"] = "C" if val <= 40 else "F"
    return Command("set_temperature", slots)


def _set_alarm(t):
    # Triggers: alarm / wake me up / get me up / wake up (at X am/pm).
    if not re.search(r"\b(alarm|wake|get me up|get up)\b", t):
        return None
    hour = _extract_int(t, ["am", "pm"])
    if hour is None:
        # An alarm without a time is not a valid command (all 31 alarm
        # commands carry a time), so reject it ("wake me up gently").
        return None
    mer = "PM" if re.search(r"\bpm\b", t) else "AM"
    # Canonical slot format matches the manifest gold: "6:00 AM".
    return Command("set_alarm", {"time": f"{hour}:00 {mer}"})


def _set_timer(t):
    # Triggers: timer / countdown (for X seconds/minutes/hours).
    if not re.search(r"\b(timer|countdown)\b", t):
        return None
    slots = {}
    mins = _extract_int(t, ["minutes", "minute", "mins", "min"])
    hrs = _extract_int(t, ["hours", "hour", "hrs", "hr"])
    secs = _extract_int(t, ["seconds", "second", "secs", "sec"])
    # Canonical slot format matches the manifest gold: "1m", "30s", "10s".
    if mins is not None:
        slots["duration"] = f"{mins}m"
    if hrs is not None:
        slots["duration"] = f"{hrs}h"
    if secs is not None:
        slots["duration"] = f"{secs}s"
    return Command("set_timer", slots)


def _lights_adjust(t):
    # Triggers: light(s) / brightness / a bare "color X" (color commands often
    # omit "light").
    has_light = re.search(r"\blight(s)?\b|\bbrightness\b", t)
    has_color_word = re.search(r"\bcolou?r(s|ed)?\b", t)
    if not (has_light or has_color_word):
        return None
    # must be an *adjust* cue, not a plain on/off
    has_dim = re.search(r"\bdim\b", t)
    has_pct = re.search(r"\bpercent\b|\b%\b", t)
    has_bright = re.search(r"\bbrightness\b", t)
    color = _first_of(t, ss.COLORS)
    is_onoff = re.search(r"\bturn (on|off)\b|\blights? (on|off)\b", t)
    if not (has_dim or has_pct or has_bright or color) or is_onoff:
        return None
    slots = {}
    pct = _extract_int(t, ["percent", "%"])
    if pct is not None:
        slots["percent"] = pct
    if color:
        slots["color"] = color
    loc = _first_of(t, ss.LOCATIONS)
    if loc:
        slots["location"] = loc
    return Command("lights_adjust", slots)


def _lights_switch(t):
    # Triggers: light(s) / lamp(s) / switch + on/off (also "kill the lights").
    if not re.search(r"\blight(s)?\b|\blamp(s)?\b", t):
        return None
    if re.search(r"\b(off|kill|shut|out)\b", t):
        state = "off"
    elif re.search(r"\b(on|up)\b", t):
        state = "on"
    else:
        return None
    slots = {"state": state}
    loc = _first_of(t, ss.LOCATIONS)
    if loc:
        slots["location"] = loc
    return Command("lights_switch", slots)


def _call(t):
    if not re.search(r"\b(call|phone|ring)\b", t):
        return None
    contact = _first_of(t, ss.CONTACTS)
    if not contact:
        return Command("call", {})  # bare "call" -> call with no contact
    return Command("call", {"contact": contact})


def _reminders_lists(t):
    if not re.search(r"\b(remind|reminder|reminders|to do|shopping list|list)\b", t):
        return None
    # "remind me to X" / "reminder X" / "create a reminder to X" -> create X
    m = re.search(r"\b(remind me to|reminder|create a reminder to|add a reminder to)\b\s*(.*)$", t)
    if m:
        action = m.group(2).strip()
        action = _first_of(action, ss.REMINDER_ACTIONS) or action or "general"
        return Command("reminders_lists", {"action": action})
    if re.search(r"\b(what are my|show|list)\b.*\b(reminders?|list)\b", t) or \
       re.search(r"\b(reminders?|list)\b", t):
        return Command("reminders_lists", {"action": "list"})
    return Command("reminders_lists", {"action": "general"})


def _media_control(t):
    table = [
        ("pause", r"\bpause\b"),
        ("stop", r"\bstop\b"),
        ("next", r"\b(next|skip)\b"),
        ("previous", r"\bprevious\b"),
        ("mute", r"\b(mute|unmute)\b"),
        ("volume_up", r"\blouder\b|\bmax\b|\bvolume (?:up|higher|high)\b"
                      r"|\b(?:increase|raise|tune|crank|boost) (?:the )?(?:sound|audio|volume|speaker)\b"
                      r"|\bturn (?:the |it |my )?(?:sound|volume)? ?up\b"
                      r"|\bturn (?:the |it |my )?(?:sound|volume)? ?high\b"
                      r"|\bneed volume\b|\btoo quiet\b|\bcan'?t hear\b"),
        ("volume_down", r"\bquieter\b|\bsofter\b|\bvolume (?:down|lower|low)\b"
                        r"|\b(?:lower|decrease|reduce|drop|cut) (?:the )?(?:sound|audio|volume|speaker|levels?)\b"
                        r"|\bturn (?:the |it |my )?(?:sound|volume)? ?down\b"
                        r"|\btoo loud\b"),
    ]
    for action, pat in table:
        if re.search(pat, t):
            return Command("media_control", {"action": action})
    return None


def _message(t):
    if re.search(r"\bmessage\b", t):
        return Command("call", {"action": "message"})
    return None


# Rule order matters: media_control before lights_switch so "turn off the
# music" (STOP) is not swallowed by the lights rule; reminders before call so
# "remind me to call mom" is a reminder, not a call.
_RULES = [
    _set_temperature, _set_alarm, _set_timer,
    _lights_adjust,
    _media_control, _lights_switch,
    _reminders_lists, _message, _call, _play_music, _ask_question,
]


def parse(text: str) -> Command:
    """Parse a transcript into a Command. Returns intent='unknown' on no match."""
    t = _normalize(text)
    if not t:
        return Command(UNKNOWN)
    for rule in _RULES:
        cmd = rule(t)
        if cmd is not None:
            return cmd
    return Command(UNKNOWN)


if __name__ == "__main__":
    samples = [
        "play jazz",
        "turn on the living room lights",
        "dim the lights to fifty percent",
        "set a timer for 5 minutes",
        "set an alarm for 7 am",
        "set the temperature to 22 degrees",
        "what's the weather",
        "pause",
        "remind me to buy milk",
        "call mom",
        "hello there",
    ]
    for s in samples:
        print(f"{s!r:40} -> {parse(s).to_dict()}")
