import os
from fastmcp import FastMCP

from tts_server.synthesize import synthesize_path

# Initialize FastMCP server
mcp = FastMCP("TTS-Server")


@mcp.tool()
async def synthesize_speech(text: str, output_path: str) -> str:
    """
    Converts text to speech using local Piper TTS and saves it to a file.

    Args:
        text: The text to synthesize.
        output_path: The file path where the audio should be saved.
    """
    audio_path, ok = synthesize_path(text, output_path)
    if not ok:
        return (
            "Error during synthesis: Piper failed or is not installed. "
            "Check PIPER_BINARY / PIPER_MODEL / PIPER_CONFIG."
        )
    return f"Audio saved successfully to {audio_path}"


if __name__ == "__main__":
    mcp.run()
