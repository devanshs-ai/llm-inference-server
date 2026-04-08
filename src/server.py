"""
Phase 5 — Production API Layer

FastAPI server with:
  - POST /generate        → blocking, returns full completion
  - POST /generate/stream → SSE streaming, tokens arrive as they're generated
  - GET  /health          → GPU stats + scheduler status
  - GET  /metrics         → benchmark numbers for completed requests
"""

import asyncio
import time
import uuid
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.schedular import ContinuousBatchingScheduler, InferenceRequest
from src.kv_cache_manager import KVCacheManager

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
KV_CACHE_GB = 3.0
MAX_BATCH   = 8

# ── Global scheduler (initialised in lifespan) ─────────────────────────────────
scheduler: ContinuousBatchingScheduler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model + start scheduler loop before accepting requests."""
    global scheduler

    print(f"Loading {MODEL_ID} on {DEVICE}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=DEVICE
    ).eval()
    print(f"Model loaded. Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    cache     = KVCacheManager(
        total_gpu_memory_gb=KV_CACHE_GB,
        eviction_policy="priority",
    )
    scheduler = ContinuousBatchingScheduler(
        model, tokenizer, cache, max_batch_size=MAX_BATCH
    )

    # Start the scheduler loop as a background task
    loop_task = asyncio.create_task(scheduler.run_forever())
    print("[SERVER] Scheduler loop started — ready to accept requests\n")

    yield   # server is live here

    loop_task.cancel()
    print("[SERVER] Scheduler loop stopped")

# ── Add run_forever back to scheduler ──────────────────────────────────────────
# (paste this method into ContinuousBatchingScheduler in scheduler.py)
#
#   async def run_forever(self):
#       while True:
#           self._admit_waiting_requests()
#           if self.active_batch:
#               self._step_sync()
#           await asyncio.sleep(0)

app = FastAPI(title="LLM Inference Server", lifespan=lifespan)

# ── Schemas ────────────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt         : str
    max_new_tokens : int = 80

class GenerateResponse(BaseModel):
    request_id      : str
    generated_text  : str
    prompt_tokens   : int
    generated_tokens: int
    ttft_sec        : float
    total_time_sec  : float
    tokens_per_sec  : float

# ── Blocking endpoint ──────────────────────────────────────────────────────────
@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    """
    Submit a request and wait for full completion.
    Under load, requests are batched by the scheduler — no explicit queuing needed.
    """
    rid   = str(uuid.uuid4())[:8]
    queue = asyncio.Queue()

    request = InferenceRequest(
        request_id    = rid,
        prompt        = req.prompt,
        max_new_tokens= req.max_new_tokens,
        output_queue  = queue,
    )
    scheduler.submit(request)

    # Collect all tokens until sentinel (None)
    tokens = []
    while True:
        token = await queue.get()
        if token is None:
            break
        tokens.append(token)

    # Find completed record for metrics
    record = next(
        (r for r in scheduler.completed_requests if r["request_id"] == rid), {}
    )

    return GenerateResponse(
        request_id      = rid,
        generated_text  = "".join(tokens),
        prompt_tokens   = record.get("prompt_tokens", 0),
        generated_tokens= record.get("generated_tokens", 0),
        ttft_sec        = record.get("ttft_sec", 0),
        total_time_sec  = record.get("total_time_sec", 0),
        tokens_per_sec  = record.get("tokens_per_sec", 0),
    )

# ── Streaming endpoint ─────────────────────────────────────────────────────────
@app.post("/generate/stream")
async def generate_stream(req: GenerateRequest):
    """
    SSE streaming — tokens are sent to the client as they're generated.
    The browser (or curl) sees words appearing in real time.
    """
    rid   = str(uuid.uuid4())[:8]
    queue = asyncio.Queue()

    request = InferenceRequest(
        request_id    = rid,
        prompt        = req.prompt,
        max_new_tokens= req.max_new_tokens,
        output_queue  = queue,
    )
    scheduler.submit(request)

    async def token_stream():
        while True:
            token = await queue.get()
            if token is None:
                yield "data: [DONE]\n\n"
                break
            # SSE format: each message is "data: ...\n\n"
            yield f"data: {token}\n\n"

    return StreamingResponse(
        token_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control"              : "no-cache",
            "X-Accel-Buffering"          : "no",   # disables nginx buffering if behind proxy
            "Access-Control-Allow-Origin": "*",
        },
    )

# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    free, total = torch.cuda.mem_get_info()
    status      = scheduler.batch_status()
    return {
        "status"          : "ok",
        "device"          : torch.cuda.get_device_name(0),
        "vram_free_gb"    : round(free  / 1e9, 2),
        "vram_total_gb"   : round(total / 1e9, 2),
        "vram_used_gb"    : round((total - free) / 1e9, 2),
        "active_requests" : status["active"],
        "waiting_requests": status["waiting"],
        "completed_total" : status["completed"],
        "scheduler_steps" : status["step"],
        "cache_utilization": status["cache"]["utilization_pct"],
    }

# ── Metrics ────────────────────────────────────────────────────────────────────
@app.get("/metrics")
async def metrics():
    completed = scheduler.completed_requests
    if not completed:
        return {"message": "No completed requests yet"}

    ttfts      = [r["ttft_sec"]       for r in completed]
    latencies  = [r["total_time_sec"] for r in completed]
    tps_vals   = [r["tokens_per_sec"] for r in completed]

    return {
        "completed_requests": len(completed),
        "avg_ttft_sec"      : round(sum(ttfts)     / len(ttfts), 4),
        "avg_latency_sec"   : round(sum(latencies)  / len(latencies), 4),
        "avg_tokens_per_sec": round(sum(tps_vals)   / len(tps_vals), 2),
        "p95_latency_sec"   : round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
        "recent"            : completed[-5:],   # last 5 for quick inspection
    }