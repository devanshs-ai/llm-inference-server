"""
Test the scheduler directly — no FastAPI, no HTTP.
Submits N requests and runs the scheduling loop until all complete.
This isolates the scheduling logic from networking concerns.
"""
import asyncio
import time
import uuid
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.schedular import ContinuousBatchingScheduler, InferenceRequest
from src.kv_cache_manager import KVCacheManager

MODEL_ID = "C:/Users/Devansh/holeeshit/llm-inference-server/models/TinyLlama-1.1B-Chat"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

TEST_PROMPTS = [
    "Explain what a neural network is in simple terms:",
    "What is gradient descent and why does it matter?",
    "Describe the attention mechanism in transformers:",
    "What is the difference between supervised and unsupervised learning?",
    "Explain what a vector database is:",
    "What is the KV cache in LLM inference?",
    "Describe how tokenization works in language models:",
    "What is CUDA and why do LLMs need GPUs?",
]

async def run_test(num_requests: int, max_new_tokens: int = 60):
    print(f"\n{'='*60}")
    print(f"Scheduler test — {num_requests} concurrent requests")
    print(f"{'='*60}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    ).eval()
    print(f"Model loaded. Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    # Init cache and scheduler
    cache   = KVCacheManager(total_gpu_memory_gb=2.0, eviction_policy="priority")
    sched   = ContinuousBatchingScheduler(model, tokenizer, cache, max_batch_size=8)

    # Create requests
    requests = []
    for i in range(num_requests):
        req = InferenceRequest(
            request_id    = f"req_{i:02d}",
            prompt        = TEST_PROMPTS[i % len(TEST_PROMPTS)],
            max_new_tokens= max_new_tokens,
            output_queue  = asyncio.Queue(),
        )
        requests.append(req)
        sched.submit(req)

    print(f"\nSubmitted {num_requests} requests to waiting queue")
    print(f"Waiting queue depth: {len(sched.waiting_queue)}")

    # Run scheduler until all requests complete
    t_start = time.perf_counter()

    async def run_until_done():
        while len(sched.completed_requests) < num_requests:
            sched._admit_waiting_requests()
            if sched.active_batch:
                sched._step_sync()          # no await — synchronous hot path
            await asyncio.sleep(0)          # yield once per full step, not per request

            if sched.total_steps % 10 == 0:
                status = sched.batch_status()
                print(f"  Step {sched.total_steps:4d} | "
                      f"active={status['active']} | "
                      f"waiting={status['waiting']} | "
                      f"done={status['completed']}/{num_requests} | "
                      f"cache={status['cache']['utilization_pct']}%")
    await run_until_done()
    t_end = time.perf_counter()

    # Results
    total_wall   = t_end - t_start
    total_tokens = sum(r["generated_tokens"] for r in sched.completed_requests)
    throughput   = total_tokens / total_wall
    avg_ttft     = sum(r["ttft_sec"] for r in sched.completed_requests) / num_requests
    avg_total    = sum(r["total_time_sec"] for r in sched.completed_requests) / num_requests

    print(f"\n{'─'*60}")
    print(f"Results ({num_requests} requests, max_tokens={max_new_tokens}):")
    print(f"  Wall time         : {total_wall:.2f}s")
    print(f"  Total tokens gen  : {total_tokens}")
    print(f"  Throughput        : {throughput:.1f} tokens/sec")
    print(f"  Avg TTFT          : {avg_ttft:.3f}s")
    print(f"  Avg total latency : {avg_total:.3f}s")
    print(f"  Scheduler steps   : {sched.total_steps}")
    print(f"  Cache peak usage  : {cache.stats['peak_blocks']} blocks")
    print(f"  Evictions         : {cache.stats['evictions']}")

    return {
        "num_requests" : num_requests,
        "wall_time"    : round(total_wall, 3),
        "throughput"   : round(throughput, 2),
        "avg_ttft"     : round(avg_ttft, 4),
        "avg_latency"  : round(avg_total, 4),
    }

async def main():
    # Test with increasing concurrency — same progression as naive benchmark
    results = []
    for n in [1, 2, 4, 8]:
        r = await run_test(num_requests=n, max_new_tokens=60)
        results.append(r)
        torch.cuda.empty_cache()
        await asyncio.sleep(1)

    # Side-by-side comparison table
    print(f"\n{'='*60}")
    print("THROUGHPUT COMPARISON (tokens/sec):")
    print(f"{'='*60}")
    print(f"{'Concurrency':<15} {'Continuous Batch':>18} {'Naive (Phase 1)':>18}")
    print(f"{'─'*52}")

    # Your Phase 1 numbers
    naive = {1: 22.17, 2: 22.35, 4: 30.96, 8: 50.60}

    for r in results:
        n     = r["num_requests"]
        cb    = r["throughput"]
        naive_tps = naive.get(n, 0)
        delta = ((cb - naive_tps) / naive_tps) * 100
        print(f"{n:<15} {cb:>18.1f} {naive_tps:>18.1f}   ({delta:+.0f}%)")

if __name__ == "__main__":
    asyncio.run(main())