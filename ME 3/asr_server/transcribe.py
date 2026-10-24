"""
Plain (non-MCP) ASR helpers.

These functions are shared by:
  * the MCP ASR server (asr_server/server.py) and
  * the full-duplex orchestrator (agent_orchestrator.py)

Keeping them free of any MCP imports means the orchestrator can use ASR
without pulling the MCP framework into its process.
"""

import os
import tempfile
from typing import Optional, Union

# Lazily-created OpenAI client so importing this module never requires the
# `openai` package or an API key (it's only needed when actually transcribing).
_client = None


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client


def transcribe_bytes(
    audio_data: bytes,
    model: str = "whisper-1",
    filename: str = "utterance.wav",
) -> str:
    """
    Transcribe raw audio bytes to text using OpenAI Whisper.

    Args:
        audio_data: Raw audio bytes (e.g. a 16 kHz mono PCM .wav).
        model: Whisper model name.
        filename: Filename used for the upload (helps Whisper pick a codec).

    Returns:
        The transcript text.

    Raises:
        RuntimeError: If transcription fails (so callers can distinguish a
            real error from an empty transcript).
    """
    if not audio_data:
        return ""

    client = _get_client()
    try:
        # Whisper accepts bytes directly; a named file object lets it infer the format.
        file_obj = _BytesFile(audio_data, filename)
        transcript = client.audio.transcriptions.create(model=model, file=file_obj)
        return transcript.text
    except Exception as e:
        raise RuntimeError(f"Transcription failed: {e}") from e


def transcribe_path(audio_path: str, model: str = "whisper-1") -> str:
    """
    Transcribe an audio file on disk to text using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file.
        model: Whisper model name.

    Returns:
        The transcript text.

    Raises:
        RuntimeError: If transcription fails.
    """
    with open(audio_path, "rb") as f:
        audio_data = f.read()
    return transcribe_bytes(audio_data, model=model, filename=os.path.basename(audio_path))


def write_temp_wav(audio_data: bytes, suffix: str = ".wav") -> str:
    """
    Write raw audio bytes to a temporary file and return its path.

    Useful when a downstream component expects a file path rather than bytes.
    The caller is responsible for deleting the file when done.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(audio_data)
    return path


class _BytesFile:
    """Minimal in-memory file object for OpenAI's multipart upload."""

    def __init__(self, data: bytes, filename: str):
        self._data = data
        self._filename = filename
        self._pos = 0

    @property
    def name(self):
        return self._filename

    def read(self, size: Optional[int] = None) -> bytes:
        if size is None:
            chunk = self._data[self._pos:]
            self._pos = len(self._data)
        else:
            chunk = self._data[self._pos:self._pos + size]
            self._pos += len(chunk)
        return chunk

    def seek(self, offset: int, whence: int = 0):
        self._pos = offset
        return self._pos

    def seekable(self) -> bool:
        return True

    def close(self):
        pass
