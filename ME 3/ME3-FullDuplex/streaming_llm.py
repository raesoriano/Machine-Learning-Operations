"""
Streaming LLM for ME3-FullDuplex.

Streams chat completions and yields *sentences* (not tokens) so the TTS
side can start speaking the first sentence while the model is still
generating the rest. This is what removes the "pause to think" feel — the
agent starts talking ~1-2 tokens after the first token arrives.

No new dependencies: uses the `openai` package already required by ME 3.
The `openai` SDK is backend-agnostic — we point it at whichever
OpenAI-compatible endpoint you want:

    LLM_BACKEND=ollama-cloud   (default)  Ollama's cloud models
        base_url  https://ollama.com/v1
        api_key   $OLLAMA_API_KEY
        model     $OLLAMA_MODEL  (default gemma4:31b)
        → weights stream from Ollama's cloud; NOTHING is downloaded to the Pi.

    LLM_BACKEND=ollama-local   a local Ollama server (e.g. on the Pi)
        base_url  http://localhost:11434/v1/
        api_key   "ollama" (required by the SDK but ignored)
        model     $OLLAMA_MODEL  (default gemma4:31b)
        → model must be pulled locally (uses Pi storage).

    LLM_BACKEND=openai         OpenAI directly (the original ME 3 default)
        base_url  https://api.openai.com/v1
        api_key   $OPENAI_API_KEY
        model     $LLM_MODEL  (default gpt-4o-mini)

All three expose the same `chat.completions.create(stream=True)` shape, so
the sentence-splitting loop below is identical regardless of backend.
"""

import os
import re
from typing import Generator, Optional

# Lazily-created client so importing this module never requires the
# `openai` package or an API key (only needed when actually calling).
_client = None


def _get_client():
    """
    Build (and cache) an OpenAI-compatible client for the active backend.

    The `openai` SDK works with any OpenAI-compatible server — we just set
    base_url + api_key per backend. Cloud Ollama needs $OLLAMA_API_KEY;
    local Ollama ignores the key; OpenAI needs $OPENAI_API_KEY.
    """
    global _client
    if _client is None:
        import openai

        backend = os.environ.get("LLM_BACKEND", "ollama-cloud").strip().lower()

        if backend == "ollama-cloud":
            base_url = "https://ollama.com/v1"
            api_key = os.environ.get("OLLAMA_API_KEY", "")
            model = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
        elif backend == "ollama-local":
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
            api_key = "ollama"  # required by the SDK, ignored by Ollama
            model = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
        elif backend == "openai":
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        else:
            raise RuntimeError(
                f"Unknown LLM_BACKEND '{backend}'. "
                "Use one of: ollama-cloud, ollama-local, openai."
            )

        if not api_key:
            hint = {
                "ollama-cloud": "set OLLAMA_API_KEY (create one at ollama.com)",
                "openai": "set OPENAI_API_KEY",
            }.get(backend, "")
            raise RuntimeError(
                f"LLM backend '{backend}' is missing its API key. {hint}"
            )

        _client = openai.OpenAI(base_url=base_url, api_key=api_key)
        # Stash the resolved model name so stream_sentences can default to it.
        _client._me3_model = model  # type: ignore[attr-defined]
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
    model: Optional[str] = None,
    max_tokens: int = 300,
    overflow_chars: int = 200,
) -> Generator[str, None, str]:
    """
    Stream the model's reply and yield sentences as they complete.

    Args:
        prompt: The user's transcribed utterance.
        system: System prompt (OnIt persona).
        model:  Model name. If None, uses the model resolved from the
                active LLM_BACKEND (see _get_client).
        max_tokens: Cap on reply length.
        overflow_chars: Flush the buffer once it reaches this length even
                without a sentence boundary (guards against unpunctuated
                output starving the TTS).

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
    if model is None:
        model = getattr(client, "_me3_model", "gpt-4o-mini")
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
