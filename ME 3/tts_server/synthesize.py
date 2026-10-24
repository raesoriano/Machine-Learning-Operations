"""
Plain (non-MCP) TTS helpers.

These functions are shared by:
  * the MCP TTS server (tts_server/server.py) and
  * the full-duplex orchestrator (agent_orchestrator.py)

Keeping them free of any MCP imports means the orchestrator can use TTS
without pulling the MCP framework into its process.

All Piper paths are configurable via environment variables so the code is
portable and no longer hard-coded to a single user's home directory.
"""

import os
import subprocess
import tempfile
from typing import Optional, Tuple

# Piper configuration — overridable via environment variables.
# Defaults point at the conventional Raspberry Pi install location.
PIPER_BINARY = os.environ.get("PIPER_BINARY", "/home/raesoriano/piper/piper")
PIPER_MODEL = os.environ.get("PIPER_MODEL", "/home/raesoriano/piper/en_US-lessac-medium.onnx")
PIPER_CONFIG = os.environ.get(
    "PIPER_CONFIG", "/home/raesoriano/piper/en_US-lessac-medium.onnx.json"
)
PIPER_LD_LIBRARY_PATH = os.environ.get("PIPER_LD_LIBRARY_PATH", "/home/raesoriano/piper")


def synthesize_bytes(
    text: str,
    output_path: Optional[str] = None,
    piper_binary: str = PIPER_BINARY,
    model_path: str = PIPER_MODEL,
    config_path: str = PIPER_CONFIG,
) -> Tuple[str, bool]:
    """
    Convert text to speech using local Piper TTS and save it to a file.

    Args:
        text: The text to synthesize.
        output_path: Where to write the .wav. If None, a temp file is created.
        piper_binary: Path to the piper executable.
        model_path: Path to the .onnx voice model.
        config_path: Path to the model's .json config.

    Returns:
        (audio_path, success) — the path to the produced .wav and whether it
        succeeded. Callers should check the boolean rather than parsing the
        returned string, which makes error handling unambiguous.
    """
    if output_path is None:
        output_path = write_temp_wav_path()

    env = dict(os.environ)
    if PIPER_LD_LIBRARY_PATH:
        env["LD_LIBRARY_PATH"] = PIPER_LD_LIBRARY_PATH

    try:
        # Piper reads text from stdin and writes the wav to --output_file.
        process = subprocess.Popen(
            [piper_binary, "--model", model_path, "--config", config_path,
             "--output_file", output_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        _, stderr = process.communicate(input=text)

        if process.returncode != 0:
            return output_path, False
        return output_path, True
    except FileNotFoundError:
        return output_path, False
    except Exception:
        return output_path, False


def synthesize_path(
    text: str,
    output_path: str,
    piper_binary: str = PIPER_BINARY,
    model_path: str = PIPER_MODEL,
    config_path: str = PIPER_CONFIG,
) -> Tuple[str, bool]:
    """Convenience wrapper that always writes to a caller-specified path."""
    return synthesize_bytes(text, output_path=output_path,
                            piper_binary=piper_binary,
                            model_path=model_path, config_path=config_path)


def write_temp_wav_path(suffix: str = ".wav") -> str:
    """Return a fresh, empty temp file path for a .wav (caller writes it)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    return path
