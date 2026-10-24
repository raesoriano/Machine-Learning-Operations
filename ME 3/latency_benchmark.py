import time
import asyncio
import numpy as np

class LatencyBenchmark:
    """
    Standardized Latency Measurement Method for Conversational AI.
    Categories:
    1. VAD Trigger Latency: Time from speech start to VAD detection.
    2. ASR Turnaround: Time from speech end to text output.
    3. LLM TTFB: Time to First Token from LLM.
    4. TTS Synthesis: Time from text input to first audio byte.
    5. Interruption Latency: Time from user speech start to TTS silence.
    """
    def __init__(self):
        self.metrics = {
            "vad_trigger": [],
            "asr_turnaround": [],
            "llm_ttfb": [],
            "tts_synthesis": [],
            "interruption": []
        }

    def record(self, category, value):
        if category in self.metrics:
            self.metrics[category].append(value)

    def report(self):
        print("\n--- Latency Benchmark Report ---")
        print(f"{'Category':<20} | {'Mean (ms)':<12} | {'P95 (ms)':<12}")
        print("-" * 48)
        for cat, values in self.metrics.items():
            if not values:
                print(f"{cat:<20} | {'N/A':<12} | {'N/A':<12}")
                continue
            mean = np.mean(values) * 1000
            p95 = np.percentile(values, 95) * 1000
            print(f"{cat:<20} | {mean:<12.2f} | {p95:<12.2f}")

async def run_benchmark_sim():
    bench = LatencyBenchmark()

    print("Simulating conversational turns... (SIMULATED — random sleep, not real pipeline)")
    for i in range(5):
        # Simulate VAD trigger
        start = time.time()
        await asyncio.sleep(0.1 + np.random.random() * 0.1)
        bench.record("vad_trigger", time.time() - start)
        
        # Simulate ASR
        start = time.time()
        await asyncio.sleep(0.3 + np.random.random() * 0.2)
        bench.record("asr_turnaround", time.time() - start)
        
        # Simulate LLM TTFB
        start = time.time()
        await asyncio.sleep(0.2 + np.random.random() * 0.3)
        bench.record("llm_ttfb", time.time() - start)
        
        # Simulate TTS
        start = time.time()
        await asyncio.sleep(0.1 + np.random.random() * 0.1)
        bench.record("tts_synthesis", time.time() - start)
        
        # Simulate Interruption
        start = time.time()
        await asyncio.sleep(0.05 + np.random.random() * 0.05)
        bench.record("interruption", time.time() - start)

    bench.report()

if __name__ == "__main__":
    asyncio.run(run_benchmark_sim())
