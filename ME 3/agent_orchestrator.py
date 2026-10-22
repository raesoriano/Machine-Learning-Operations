import asyncio
import time
import numpy as np
import pyaudio
from vad_system.vad_manager import VADManager
from asr_server.server import transcribe_audio
from tts_server.server import synthesize_speech

class FullDuplexAgent:
    def __init__(self):
        self.vad = VADManager()
        self.is_speaking = False
        self.stop_tts = False
        self.audio_stream = None
        
        # Callbacks for VAD
        self.vad.on_speech_start = self.handle_speech_start
        self.vad.on_speech_end = self.handle_speech_end

    async def handle_speech_start(self):
        print("[Agent] User started speaking... Interrupting TTS.")
        self.stop_tts = True
        if self.is_speaking:
            # In a real implementation, this would kill the audio playback process/stream
            self.is_speaking = False

    async def handle_speech_end(self):
        print("[Agent] User stopped speaking. Processing request...")
        # 1. Capture the audio buffer from VAD
        audio_data = self.vad.get_last_utterance()
        
        # 2. ASR (via MCP Server logic)
        text = await transcribe_audio(audio_data)
        print(f"[User]: {text}")
        
        # 3. LLM Processing (Simulated as a commercial API call)
        response_text = await self.call_llm(text)
        print(f"[AI]: {response_text}")
        
        # 4. TTS (via MCP Server logic)
        await self.speak(response_text)

    async def call_llm(self, text):
        # Simulated LLM call
        await asyncio.sleep(0.5) 
        return f"I heard you say '{text}'. I am your full-duplex agent."

    async def speak(self, text):
        self.stop_tts = False
        self.is_speaking = True
        
        # Get audio from ElevenLabs MCP Server
        audio_file = await synthesize_speech(text)
        
        print("[Agent] Playing response...")
        # Simulated playback that can be interrupted
        # In production, this would use a non-blocking audio player (e.g., pygame or pydub)
        # that checks self.stop_tts in a loop.
        
        # Simulation of playback duration
        playback_duration = 3.0 
        start_time = time.time()
        while time.time() - start_time < playback_duration:
            if self.stop_tts:
                print("[Agent] TTS Interrupted!")
                break
            await asyncio.sleep(0.1)
            
        self.is_speaking = False

    async def run(self):
        print("Starting Full-Duplex Agent... Press Ctrl+C to stop.")
        # Start VAD in a separate thread or async task
        vad_task = asyncio.create_task(self.vad.listen_loop())
        
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            vad_task.cancel()

if __name__ == "__main__":
    agent = FullDuplexAgent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        pass
