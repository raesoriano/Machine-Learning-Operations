"""live_demo.adapters -- one module per recognizer family.

Each adapter adds its own `pi test vN` folder to sys.path and imports the
PRODUCTION recognizer (the exact code the Pi runs in that folder), then wraps
it so it exposes the shared interface:

    classify_warm(pcm: bytes, reps: int = 1) -> (command, intent, transcript, prob)

where `command` is the COARSE 19-command schema label (or "REJECT").
"""
