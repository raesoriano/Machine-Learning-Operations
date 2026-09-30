"""Closed slot spaces — the SINGLE SOURCE OF TRUTH for the VCM command set.

Used by:
  - dataset template grammars   (data/templates/ — collective work)
  - constrained model vocabulary (model/vocab.py)
  - deterministic parser         (parser/parser.py)
  - benchmark                    (benchmark/)

Anything the model can say and anything the parser can understand derives
from the lists below. Changing a value here is a breaking change: it requires
a PR review (see COLLECTIVE_TASKS.md, WP1).
"""
from .numbers import int_to_words  # noqa: F401  (re-exported for templates)

# --- lights ---------------------------------------------------------------
LOCATIONS = ["living room", "kitchen", "bedroom", "bathroom", "office", "hallway"]
COLORS = ["red", "blue", "green", "yellow", "white", "warm", "cool"]
PERCENTS = list(range(5, 101, 5))          # 5, 10, ..., 100 ("half" -> 50 is an alias)

# --- timers / alarms / temperature -----------------------------------------
DURATION_MINUTES = list(range(1, 61)) + [90, 120]
DURATION_HOURS = list(range(1, 25))
CLOCK_HOURS = list(range(1, 13))
MERIDIEM = ["am", "pm"]
TEMP_C = list(range(16, 31))               # 16-30 Celsius
TEMP_F = list(range(60, 86))               # 60-85 Fahrenheit

# --- contacts (closed list; ~20) -------------------------------------------
CONTACTS = [
    "mom", "dad", "mama", "papa", "sister", "brother", "grandma", "grandpa",
    "aunt", "uncle", "cousin", "friend", "best friend", "neighbor", "doctor",
    "school", "work", "boss", "partner", "coworker",
]

# --- music ------------------------------------------------------------------
MUSIC_GENRES = [
    "rock", "jazz", "pop", "classical", "lofi", "country", "hip hop",
    "blues", "reggaeton", "latin",
]
MUSIC_ARTISTS = [
    "the Beatles", "Elvis Presley", "Taylor Swift", "Beyonce", "Ed Sheeran",
    "Adele", "Drake", "Dua Lipa", "The Weeknd", "Billie Eilish", "Coldplay",
    "Queen", "Michael Jackson", "Miles Davis", "Johnny Cash", "Bob Marley",
    "Ariana Grande", "Bruno Mars", "Kendrick Lamar", "Justin Bieber",
]

# --- reminder actions (closed list for generation) ---------------------------
REMINDER_ACTIONS = [
    "call mom", "buy milk", "take out the trash", "water the plants",
    "pick up the kids", "take the medicine", "feed the dog", "email the report",
    # OptionB 31-command set (ME2): reminder task values
    "drink water", "exercise", "study",
]


def phrases_to_words(phrases):
    """Expand multi-word phrases into individual words (for the vocab)."""
    words = set()
    for p in phrases:
        words.update(p.lower().split())
    return words
