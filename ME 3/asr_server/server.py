import os
from typing import Any
from mcp.server.fastmcp import FastMCP
import openai

# Initialize FastMCP server
mcp = FastMCP("ASR-Server")

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@mcp.tool()
async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes an audio file to text using OpenAI Whisper.
    
    Args:
        audio_path: Path to the audio file to be transcribed.
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file
            )
        return transcript.text
    except Exception as e:
        return f"Error during transcription: {str(e)}"

if __name__ == "__main__":
    mcp.run()
