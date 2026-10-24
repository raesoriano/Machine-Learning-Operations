import os
from mcp.server.fastmcp import FastMCP

from asr_server.transcribe import transcribe_path

# Initialize FastMCP server
mcp = FastMCP("ASR-Server")


@mcp.tool()
async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file to text using OpenAI Whisper.

    Args:
        audio_path: Path to the audio file to be transcribed.
    """
    try:
        return transcribe_path(audio_path)
    except Exception as e:
        return f"Error during transcription: {str(e)}"


if __name__ == "__main__":
    mcp.run()
