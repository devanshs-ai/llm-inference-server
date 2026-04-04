"""
Benchmark script for the naive server.
Sends N concurrent requests and measures real-world latency under load.
Run this in a SEPARATE terminal while the server is running.
"""
import asyncio
import aiohttp
import time
import json
import statistics
import pandas as pd
from datetime import datetime

SERVER_URL = "http://localhost:8000/generate"

TEST_PROMPTS = [
    "Explain what a neural network is:",
    "What is the capital of France and why is it important?",
    "Describe how photosynthesis works:",
    "What are the main causes of climate change?",
    "Explain the difference between RAM and ROM:",
]

async def single_request(session: aiohttp.ClientSession, prompt: str, request_id: int):
    payload = {"prompt": prompt, "max_new_tokens": 80}
    t_send  = time.perf_counter()
    
    try:
        async with session.post(SERVER_URL, json=payload) as resp:
            # If the server isn't happy, tell us why
            if resp.status != 200:
                error_body = await resp.text()
                print(f"Server Error (Req {request_id}): Status {resp.status}, Body: {error_body}")
                # Return a "dummy" result so the rest of the script doesn't crash
                return {"generated_tokens": 0, "total_time_sec": 0, "tokens_per_sec": 0, "wall_time_sec": 0, "prompt_tokens": 0}
            
            result = await resp.json()
    except Exception as e:
        print(f"Connection Error (Req {request_id}): {e}")
        return {"generated_tokens": 0, "total_time_sec": 0, "tokens_per_sec": 0, "wall_time_sec": 0, "prompt_tokens": 0}

    t_recv = time.perf_counter()

    # Match the keys with the 'GenerateResponse' class in your server
    return {
        "request_id"      : request_id,
        "prompt_tokens"   : result["prompt_tokens"],
        "generated_tokens": result["generated_tokens"],
        "server_time_sec" : result["total_time_sec"],
        "wall_time_sec"   : round(t_recv - t_send, 3),
        "tokens_per_sec"  : result["tokens_per_sec"],
    }

async def run_concurrent_benchmark(concurrency: int):
    """Fire `concurrency` requests simultaneously and wait for all to finish."""
    print(f"\n--- Concurrency = {concurrency} ---")
    connector = aiohttp.TCPConnector(limit=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            single_request(session, TEST_PROMPTS[i % len(TEST_PROMPTS)], i)
            for i in range(concurrency)
        ]
        t_batch_start = time.perf_counter()
        results       = await asyncio.gather(*tasks)
        t_batch_end   = time.perf_counter()

    wall_times  = [r["wall_time_sec"]   for r in results]
    tps_vals    = [r["tokens_per_sec"]  for r in results]
    batch_time  = t_batch_end - t_batch_start
    total_tokens= sum(r["generated_tokens"] for r in results)

    print(f"  Requests completed : {len(results)}")
    print(f"  Batch wall time    : {batch_time:.2f}s")
    print(f"  Throughput         : {total_tokens/batch_time:.1f} tokens/sec (batch)")
    print(f"  Avg latency/req    : {statistics.mean(wall_times):.2f}s")
    print(f"  Max latency/req    : {max(wall_times):.2f}s  ← this is what user N feels")
    print(f"  P95 latency        : {sorted(wall_times)[int(len(wall_times)*0.95)]:.2f}s")

    return {
        "concurrency"     : concurrency,
        "batch_time_sec"  : round(batch_time, 3),
        "throughput_tps"  : round(total_tokens / batch_time, 2),
        "avg_latency_sec" : round(statistics.mean(wall_times), 3),
        "max_latency_sec" : round(max(wall_times), 3),
        "p95_latency_sec" : round(sorted(wall_times)[int(len(wall_times)*0.95)], 3),
    }

async def main():
    print("=== Naive Server Benchmark ===")
    print("Make sure the server is running: uvicorn src.naive_server:app --port 8000 --workers 1")

    # Warm-up: one request to load any lazy-init paths
    print("\nWarm-up request...")
    async with aiohttp.ClientSession() as s:
        await single_request(s, "Hello", 0)
    print("Warm-up done.")

    # Run at increasing concurrency levels
    summary = []
    for concurrency in [1, 2, 4, 8]:
        row = await run_concurrent_benchmark(concurrency)
        summary.append(row)
        await asyncio.sleep(2)   # let GPU cool between runs

    # Save results
    df = pd.DataFrame(summary)
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"results/naive_benchmark_{timestamp}.csv"
    df.to_csv(output_path, index=False)

    print(f"\n=== Summary saved to {output_path} ===")
    print(df.to_string(index=False))
    print("\nThese are your BASELINE numbers. Everything in Phase 3-4 is measured against this.")

if __name__ == "__main__":
    asyncio.run(main())