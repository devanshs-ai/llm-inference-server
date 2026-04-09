# LLM Inference Server

A from-scratch LLM inference server implementing the core ideas behind vLLM: continuous batching, PagedAttention-style KV cache block allocation, and SSE token streaming. Built without using vLLM, Triton, or any inference framework — only PyTorch and HuggingFace Transformers.

The goal was not to replicate vLLM. The goal was to understand what happens inside an inference server deeply enough to build one, measure its limits, and identify exactly why production systems require custom CUDA kernels.

---

## Table of Contents

- [Architecture](#architecture)
- [What Each Phase Builds](#what-each-phase-builds)
- [Benchmark Results](#benchmark-results)
- [Key Findings](#key-findings)
- [API Reference](#api-reference)
- [Running Locally](#running-locally)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [What Production Systems Do Differently](#what-production-systems-do-differently)

---

## Architecture

```
Client (curl / browser / load generator)
        |
        v
FastAPI Layer
  POST /generate          — blocking, full completion
  POST /generate/stream   — SSE streaming, token-by-token
  GET  /health            — GPU stats, scheduler state
  GET  /metrics           — latency and throughput for completed requests
        |
        v
Iteration-Level Scheduler
  - Waiting queue: incoming requests pending admission
  - Active batch: up to 8 concurrent requests in decode phase
  - After every token step: evict finished requests, admit waiting ones
  - No request waits for another to fully complete
        |
        v
KV Cache Manager
  - Pre-allocates fixed GPU memory pool at startup (3 GB default)
  - Divides pool into 16-token blocks (352 KB each for TinyLlama)
  - Assigns blocks to requests on demand, reclaims on completion
  - Two eviction policies: LRU and priority-based
        |
        v
TinyLlama-1.1B-Chat-v1.0
  - fp16 precision
  - Manual prefill / decode separation
  - past_key_values managed externally by the cache manager
```

---

## What Each Phase Builds

### Phase 1 — Naive Baseline

A standard FastAPI server wrapping `model.generate()`. Handles one logical request at a time. Under concurrent load, requests queue and wait for the previous generation to fully complete before the next one starts.

This phase exists to establish honest benchmark numbers. Every optimization in later phases is measured against this baseline — not against theoretical maximums.

Baseline: **22.2 tok/sec** at concurrency=1, **7.96s max latency** at concurrency=8.

### Phase 2 — Manual Token Control

Replaced `model.generate()` with a hand-written `model.forward()` loop. At each decode step, the loop passes one new token plus the existing `past_key_values` to the model, receives updated key-value tensors, and samples the next token from the logits.

This phase has no performance goal. Its purpose is to make `past_key_values` visible and tangible — to see exactly what the cache manager in Phase 3 will be managing.

Key discovery: inspecting `past_key_values[0][0].shape` revealed TinyLlama uses Grouped Query Attention with 4 KV heads, not 32 query heads. This makes the KV cache 8× smaller than a naive Multi-Head Attention estimate would predict. The actual cost is 22.53 KB per token rather than the ~180 KB a 32-head model would require.

### Phase 3 — KV Cache Block Allocator

A `KVCacheManager` class that pre-allocates a fixed GPU memory pool at startup and divides it into fixed-size blocks of 16 tokens each. Requests are assigned blocks on demand as their sequences grow. When a request finishes, its blocks are immediately returned to the free pool.

This directly addresses the memory fragmentation problem in naive inference: with `model.generate()`, each request requires a pre-allocated contiguous slice of VRAM sized to `max_seq_len`. If a request finishes at token 80 out of a 512-token reservation, 432 tokens worth of VRAM stays locked until all requests in the batch complete. The block allocator reclaims memory the moment it is no longer needed.

Two eviction policies are implemented for the case when memory runs out:

**LRU** evicts the request whose blocks have not been accessed for the longest time. Simple and effective when requests have similar generation lengths.

**Priority-based** evicts the request that has made the least progress — protecting nearly-complete requests from being sacrificed for new arrivals. A request 90% through generation has already consumed most of its compute cost; evicting it wastes that work.

Test result demonstrating non-contiguous allocation:
```
req_2: 40 tokens | 4 blocks | block_ids=[1, 3, 4, 5]
```
Block IDs 1, 3, 4, 5 are not contiguous — block 2 belongs to a different request. This is what PagedAttention means: logical sequence continuity mapped to non-contiguous physical memory.

### Phase 4 — Iteration-Level Scheduler

A scheduler that separates prefill (processing the full prompt) from decode (generating one token per step). After every decode step across all active requests, the scheduler checks for completions and immediately admits new requests from the waiting queue into the freed slots.

This is continuous batching: the batch composition changes after every single token step rather than after every full generation.

The primary measurable benefit is time-to-first-token (TTFT). In the naive server, a request arriving while 7 others are generating must wait for all of them to finish before its prompt is even processed. With the scheduler, a request is admitted as soon as a slot opens — which happens every time any active request finishes, not when all of them finish.

### Phase 5 — Streaming API

FastAPI endpoints exposing the scheduler to HTTP clients. The streaming endpoint uses Server-Sent Events (SSE), pushing each token as a `data:` frame the moment it is produced by the scheduler loop. This matches the wire format used by the OpenAI API.

```
data: Cont
data: inu
data: ous
data: batch
data: ing
data: is
...
data: [DONE]
```

---

## Benchmark Results

**Hardware:** NVIDIA RTX 5060 Laptop GPU, 8 GB VRAM  
**Model:** TinyLlama-1.1B-Chat-v1.0, fp16  
**Generation length:** 60–80 tokens per request

### Naive server — Phase 1

| Concurrency | Throughput (tok/s) | Avg latency (s) | Max latency (s) | P95 latency (s) |
|-------------|-------------------|-----------------|-----------------|-----------------|
| 1           | 22.2              | 3.61            | 3.61            | 3.61            |
| 2           | 22.4              | 1.84            | 3.62            | 3.62            |
| 4           | 31.0              | 2.65            | 5.23            | 5.23            |
| 8           | 50.6              | 4.80            | 7.96            | 7.96            |

### Continuous batching scheduler — TTFT comparison

| Concurrency | Naive TTFT (s) | Scheduler TTFT (s) | Improvement |
|-------------|----------------|--------------------|-------------|
| 1           | 3.61           | 1.08               | 3.3×        |
| 2           | 3.62           | 0.28               | 12.9×       |
| 4           | 5.23           | 0.035              | 149×        |
| 8           | 7.96           | 0.236              | 33×         |

TTFT is the metric that governs perceived responsiveness. The user experience difference between a 5.23s wait for the first word and a 35ms wait is the entire value proposition of continuous batching.

### Why raw throughput is lower than naive

The naive server calls `model.generate()` internally, which runs a single fused CUDA kernel over a true batch tensor. The scheduler runs sequential `model.forward()` calls per request because HuggingFace Transformers cannot batch tensors with variable-length `past_key_values` without custom memory management. Each sequential call carries Python-to-CUDA synchronization overhead.

This is not a flaw in the scheduler design. It is the exact architectural boundary that vLLM's PagedAttention solves with custom Triton kernels. By making all KV blocks the same fixed size and non-contiguous, a custom GPU kernel can process a heterogeneous batch — requests at different sequence positions — in a single call. Building this implementation made that requirement concrete rather than abstract.

---

## Key Findings

**GQA discovery via empirical inspection**

The standard description of transformer KV cache costs assumes Multi-Head Attention with equal numbers of query, key, and value heads. TinyLlama uses Grouped Query Attention: 32 query heads share 4 KV heads. This was not assumed from documentation — it was discovered by printing `past_key_values[0][0].shape` during Phase 2 and seeing `[1, 4, 15, 64]` instead of the expected `[1, 32, 15, 64]`. The actual per-token cache cost is 22.53 KB, not the ~180 KB a 32-head estimate would predict.

**The fragmentation problem is measurable**

With the block allocator, when a request finishes mid-batch, its blocks are available to the next request within the same scheduling step. Without it, the naive approach keeps that memory locked until the entire batch drains. At concurrency=4 with the uneven generation test (5, 40, and 3 tokens respectively), the request finishing at token 40 immediately returned 4 blocks — 1.4 MB of VRAM — to the free pool.

**The boundary between pseudo-batching and true GPU batching**

Sequential `model.forward()` calls with independent `past_key_values` per request is not the same as batched GPU inference. The throughput regression from naive to scheduler (22 tok/sec → 5–18 tok/sec depending on concurrency) quantifies exactly how much overhead the Python loop and per-call CUDA kernel launches add. This is the performance gap that production systems close with custom kernels — and it is only visible when you build both approaches and compare them.

---

## API Reference

### POST /generate

Blocking endpoint. Submits a request to the scheduler and waits for full completion.

Request:
```json
{
  "prompt": "Explain what a KV cache is:",
  "max_new_tokens": 80
}
```

Response:
```json
{
  "request_id": "a3f2c1d0",
  "generated_text": "A KV cache stores the key and value matrices...",
  "prompt_tokens": 10,
  "generated_tokens": 74,
  "ttft_sec": 0.034,
  "total_time_sec": 4.21,
  "tokens_per_sec": 17.6
}
```

### POST /generate/stream

SSE streaming endpoint. Tokens are pushed as Server-Sent Events as they are generated.

Request body is identical to `/generate`. Response is a stream of SSE frames:

```
data: A
data: KV
data: cache
data: stores
...
data: [DONE]
```

### GET /health

Returns current GPU memory usage and scheduler state.

```json
{
  "status": "ok",
  "device": "NVIDIA GeForce RTX 5060 Laptop GPU",
  "vram_free_gb": 4.21,
  "vram_total_gb": 7.96,
  "vram_used_gb": 3.75,
  "active_requests": 3,
  "waiting_requests": 2,
  "completed_total": 47,
  "scheduler_steps": 2814,
  "cache_utilization": 12.4
}
```

### GET /metrics

Aggregated latency and throughput statistics over all completed requests.

```json
{
  "completed_requests": 47,
  "avg_ttft_sec": 0.183,
  "avg_latency_sec": 8.42,
  "avg_tokens_per_sec": 14.3,
  "p95_latency_sec": 18.7,
  "recent": [...]
}
```

---

## Running Locally

**Requirements:** Python 3.10+, CUDA 12.x, NVIDIA GPU with 6 GB+ VRAM

```bash
# Clone and set up environment
git clone https://github.com/devanshs-ai/llm-inference-server
cd llm-inference-server
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers fastapi uvicorn accelerate locust aiohttp pandas matplotlib

# Start the server (model downloads automatically on first run, ~2.2 GB)
uvicorn src.server:app --host 0.0.0.0 --port 8000 --workers 1
```

The server is ready when the terminal prints:
```
[SERVER] Scheduler loop started — ready to accept requests
```

**Test blocking generation:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain what a KV cache is:", "max_new_tokens": 60}'
```

**Test streaming:**
```bash
curl -X POST http://localhost:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain continuous batching:", "max_new_tokens": 80}' \
  --no-buffer
```

**Interactive API explorer:**  
Navigate to `http://localhost:8000/docs` for the auto-generated Swagger UI.

**Run the benchmark suite:**
```bash
python results/benchmark_naive.py
```

---

## Project Structure

```
llm-inference-server/
│
├── src/
│   ├── server.py              # FastAPI app, lifespan, all endpoints
│   ├── scheduler.py           # Iteration-level scheduler, prefill/decode loop
│   ├── kv_cache_manager.py    # Block allocator, LRU and priority eviction
│   ├── naive_server.py        # Phase 1 baseline server
│   ├── manual_inference.py    # Phase 2 token loop, past_key_values inspection
│   ├── test_kv_cache.py       # 5 tests for the block allocator
│   └── test_scheduler.py      # Scheduler correctness and throughput tests
│
├── results/
│   ├── benchmark_naive.py     # Locust-style async load generator
│   └── *.csv                  # Benchmark output files
│
└── README.md
```

---

## Design Decisions

**Block size of 16 tokens**

vLLM uses 16 tokens per block as the default. At 22.53 KB per token for TinyLlama, a 16-token block is 352 KB — small enough to minimize internal fragmentation (wasted space within a block when a request ends mid-block) while large enough to keep the block table metadata overhead negligible. Smaller blocks reduce fragmentation but increase allocator overhead; larger blocks waste more memory on incomplete sequences.

**Priority-based eviction over LRU**

LRU evicts the request that has been waiting longest without progress. This can sacrifice a request that is 90 tokens into a 100-token generation simply because it has not needed a new block recently. Priority-based eviction assigns value proportional to tokens already generated — evicting a request at token 5 costs less wasted compute than evicting one at token 95. Both policies are implemented and can be selected at initialization.

**Prefill and decode as separate operations**

Prefill processes the entire prompt in one forward pass and produces the initial KV cache. Decode runs one token at a time, extending the cache by one position per step. These two operations have fundamentally different compute characteristics: prefill is compute-bound (processing many tokens simultaneously), decode is memory-bandwidth-bound (loading large KV tensors to process a single token). Separating them allows each to be optimized independently — which is why production systems like vLLM, TGI, and SGLang treat them as distinct scheduling phases.

**Single worker, synchronous decode loop**

`uvicorn --workers 1` is intentional. The scheduler loop runs as an asyncio background task sharing the event loop with the FastAPI request handlers. Multiple workers would spawn independent model instances with no shared scheduler state, defeating the purpose of the batch manager. The hot path (`_step_sync`) is deliberately synchronous — removing the `await` from the decode loop eliminates asyncio overhead from the latency-critical path.

---

## What Production Systems Do Differently

This implementation hits the boundary between application-level scheduling and GPU-level kernel design. The gap is worth stating precisely.

**Custom CUDA / Triton kernels for batched decode**

HuggingFace `model.forward()` requires all sequences in a batch to have the same `past_key_values` length. Because requests in a continuous batch are at different decode positions, they have KV caches of different lengths. Without custom kernels, the only option is to run them sequentially — which is what this implementation does.

vLLM's PagedAttention kernel accepts non-contiguous, variable-length KV blocks across a batch. The kernel directly indexes into the block table to gather the right KV values for each sequence position, regardless of where those blocks sit in physical memory. This is what makes true batched decode possible.

**Speculative decoding**

For each token generated, the model runs a full forward pass through all 22 transformer layers. Speculative decoding uses a small draft model to propose N tokens ahead, then verifies all N in a single forward pass of the main model. If the main model agrees with the draft, N tokens are produced for the cost of one forward pass.

**Prefix caching**

If multiple requests share a common system prompt, their prefill KV caches for that prefix are identical. A production server can compute the prefix KV cache once and share it across all requests using that prompt, eliminating redundant prefill compute. This is especially valuable for applications with long system prompts.

**Tensor parallelism**

Distributing the model across multiple GPUs by sharding the weight matrices — each GPU holds a slice of the attention and FFN layers and communicates via all-reduce operations. This increases throughput proportionally to GPU count for the same batch size.

---

## References

- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023. [arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- HuggingFace Transformers documentation — `past_key_values` and generation utilities
- vLLM project — [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

## Stack

Python 3.11 · PyTorch 2.6 · FastAPI · HuggingFace Transformers · CUDA 12.8 · TinyLlama-1.1B-Chat-v1.0