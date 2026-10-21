import os
from typing import Any
from mcp.server.fastmcp import FastMCP
import requests

# Initialize FastMCP server
mcp = FastMCP("TTS-Server")

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY")
VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Default Rachel voice

@mcp.tool()
async def synthesize_speech(text: str, output_path: str) -> str:
    """
    Converts text to speech using ElevenLabs and saves it to a file.
    
    Args:
        text: The text to synthesize.
        output_path: The file path where the audio should be saved.
    """
    if not ELEVENLABS_API_KEY:
        return "Error: ELEVENLABS_API_KEY not set in environment."

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    data = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)
        return f"Audio saved successfully to {output_path}"
    except Exception as e:
        return f"Error during synthesis: {str(e)}"

if __name__ == "__main__":
    mcp.run()
