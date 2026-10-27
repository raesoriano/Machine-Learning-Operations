import time
import asyncio
import numpy as np

from telemetry import default as telemetry, CATEGORIES


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
        self.metrics = {cat: [] for cat in CATEGORIES}

    def record(self, category, value):
        if category in self.metrics:
            self.metrics[category].append(value)

    def load_from_telemetry(self):
        """
        Populate metrics from the orchestrator's real telemetry log.
        Returns the number of latency records loaded.
        """
        records = telemetry.read_recent(max_lines=2000)
        count = 0
        for r in records:
            if r.get("type") == "latency" and r.get("category") in self.metrics:
                # Stored in ms; convert to seconds to match record().
                self.metrics[r["category"]].append(r["value_ms"] / 1000.0)
                count += 1
        return count

    def report(self, source="telemetry"):
        print(f"\n--- Latency Benchmark Report ({source}) ---")
        print(f"{'Category':<20} | {'N':<6} | {'Mean (ms)':<12} | {'P95 (ms)':<12}")
        print("-" * 56)
        for cat, values in self.metrics.items():
            if not values:
                print(f"{cat:<20} | {0:<6} | {'N/A':<12} | {'N/A':<12}")
                continue
            mean = np.mean(values) * 1000
            p95 = np.percentile(values, 95) * 1000
            print(f"{cat:<20} | {len(values):<6} | {mean:<12.2f} | {p95:<12.2f}")


def run_benchmark_from_telemetry():
    """Report real latency stats from the orchestrator's telemetry log."""
    bench = LatencyBenchmark()
    count = bench.load_from_telemetry()
    if count == 0:
        print("No real telemetry found in "
              f"{telemetry.log_file}.")
        print("Start `agent_orchestrator.py` and complete at least one turn, "
              "or run `python latency_benchmark.py --simulate` for a demo.")
        return
    bench.report(source="real telemetry")


async def run_benchmark_sim():
    """
    SIMULATED benchmark — random sleeps, NOT the real pipeline. Use this only
    to sanity-check the reporting format. For real numbers, run without
    --simulate once the agent has produced telemetry.
    """
    bench = LatencyBenchmark()
    print("Simulating conversational turns... "
          "(SIMULATED — random sleep, not the real pipeline)")
    for _ in range(5):
        start = time.time()
        await asyncio.sleep(0.1 + np.random.random() * 0.1)
        bench.record("vad_trigger", time.time() - start)

        start = time.time()
        await asyncio.sleep(0.3 + np.random.random() * 0.2)
        bench.record("asr_turnaround", time.time() - start)

        start = time.time()
        await asyncio.sleep(0.2 + np.random.random() * 0.3)
        bench.record("llm_ttfb", time.time() - start)

        start = time.time()
        await asyncio.sleep(0.1 + np.random.random() * 0.1)
        bench.record("tts_synthesis", time.time() - start)

        start = time.time()
        await asyncio.sleep(0.05 + np.random.random() * 0.05)
        bench.record("interruption", time.time() - start)

    bench.report(source="simulated")


if __name__ == "__main__":
    import sys
    if "--simulate" in sys.argv:
        asyncio.run(run_benchmark_sim())
    else:
        run_benchmark_from_telemetry()
