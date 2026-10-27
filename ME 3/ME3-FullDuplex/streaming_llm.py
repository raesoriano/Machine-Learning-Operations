"""
Streaming LLM for ME3-FullDuplex.

Streams chat completions from OpenAI and yields *sentences* (not tokens)
so the TTS side can start speaking the first sentence while the model is
still generating the rest. This is what removes the "pause to think" feel
— the agent starts talking ~1-2 tokens after the first token arrives.

No new dependencies: uses the `openai` package already required by ME 3.
"""

import os
import re
from typing import Generator, Optional

# Lazily-created client so importing this module never requires the
# `openai` package or an API key (only needed when actually calling).
_client = None


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


# Sentence boundary: end-of-sentence punctuation optionally followed by
# closing quotes/parens, then whitespace. Also treat newlines as breaks.
_SENTENCE_RE = re.compile(
    r"(?<=[.!?])[\s\"'”’)\]]+|(?<=\n)"
)


def split_sentences(buffer: str):
    """
    Split a text buffer into (complete_sentences, remainder).

    A sentence is complete when it ends with . ! or ? (plus optional
    closing quotes) followed by whitespace, or on a newline. Incomplete
    trailing text is returned as the remainder.
    """
    sentences = []
    last = 0
    for m in _SENTENCE_RE.finditer(buffer):
        end = m.end()
        chunk = buffer[last:end].strip()
        if chunk:
            sentences.append(chunk)
        last = end
    remainder = buffer[last:].strip()
    return sentences, remainder


def stream_sentences(
    prompt: str,
    system: str = "You are OnIt, a helpful voice assistant. "
                  "Keep answers short and conversational — one or two "
                  "sentences unless asked for detail.",
    model: str = "gpt-4o-mini",
    max_tokens: int = 300,
    overflow_chars: int = 200,
) -> Generator[str, None, str]:
    """
    Stream the model's reply and yield sentences as they complete.

    Yields:
        One sentence (str) each time a sentence boundary is reached in the
        token stream. The first yield happens as soon as the first sentence
        is complete — typically within the first few tokens.

    Returns:
        The full reply text (for logging/telemetry).

    Raises:
        RuntimeError: If the API call fails.
    """
    client = _get_client()
    full_text = []
    buffer = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            stream=True,
        )
        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta is None or not delta.content:
                continue
            full_text.append(delta.content)
            buffer += delta.content
            # Emit any complete sentences; keep the tail in the buffer.
            sentences, buffer = split_sentences(buffer)
            for s in sentences:
                yield s
            # Overflow guard: if the model never punctuates (e.g. a long
            # list), flush the buffer once it gets long enough so TTS
            # doesn't starve.
            if len(buffer) >= overflow_chars:
                yield buffer
                buffer = ""
        # Flush whatever is left (final fragment without terminal punct).
        if buffer.strip():
            yield buffer.strip()
    except Exception as e:
        raise RuntimeError(f"LLM streaming failed: {e}") from e

    return "".join(full_text)
