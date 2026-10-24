import os
import subprocess
from typing import Any
from fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("TTS-Server")

# Configuration for Piper
# We assume the piper binary and model are installed/available in these paths
PIPER_BINARY = "/home/raesoriano/piper/piper"
# Default to a common English voice model path
# In a real deployment, the user would download the .onnx and .json files
MODEL_PATH = "/home/raesoriano/piper/en_US-lessac-medium.onnx"
CONFIG_PATH = "/home/raesoriano/piper/en_US-lessac-medium.onnx.json"

@mcp.tool()
async def synthesize_speech(text: str, output_path: str) -> str:
    """
    Converts text to speech using local Piper TTS and saves it to a file.
    
    Args:
        text: The text to synthesize.
        output_path: The file path where the audio should be saved.
    """
    try:
        # Piper takes text from stdin and writes wav to stdout or a file
        # Command: echo "text" | piper --model model.onnx --output_file out.wav
        process = subprocess.Popen(
            [PIPER_BINARY, "--model", MODEL_PATH, "--config", CONFIG_PATH, "--output_file", output_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "LD_LIBRARY_PATH": "/home/raesoriano/piper"}
        )
        stdout, stderr = process.communicate(input=text)

        if process.returncode != 0:
            return f"Error during synthesis: {stderr}"

        return f"Audio saved successfully to {output_path}"
    except FileNotFoundError:
        return "Error: Piper binary not found. Please install Piper on the system."
    except Exception as e:
        return f"Unexpected error during synthesis: {str(e)}"

if __name__ == "__main__":
    mcp.run()
