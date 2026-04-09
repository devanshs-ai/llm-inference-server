"""
Phase 4 — Iteration-Level Scheduler (corrected)

Key architectural note included in README:
  This implementation demonstrates continuous batching scheduling logic.
  Throughput is bounded by sequential HuggingFace forward passes because
  variable-length past_key_values cannot be batched without custom CUDA kernels.
  This is exactly the problem vLLM's PagedAttention + Triton kernels solve.
  The measurable win here is TTFT: requests get their first token immediately
  rather than waiting for the entire queue to drain.

  my result shows that at concurrency = 8 the naive server is more than double the throughput 
  observed at my continuous batching server, this is because, higginface's model.generate()
  sends all the 8 requests batched as one at once, for example
  its  like a big bus, everyone gets on and goes to the gpu at the same time and gets off at the same time
  its fast since it makes only one trip

  continuous batching which we implemented, fails here because we call a for loop everytime, for e
  example we have 8 individual cars this time, its much flexible, people can get in and get out at anytime, but 
  we cause a traffic jam at the gpu's entrance.
"""

import time
import asyncio
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.kv_cache_manager import KVCacheManager

MODEL_ID       = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
MAX_BATCH_SIZE = 8
MAX_NEW_TOKENS = 100

@dataclass
class InferenceRequest:
    request_id     : str
    prompt         : str
    max_new_tokens : int = MAX_NEW_TOKENS
    input_ids      : Optional[torch.Tensor] = None
    prompt_tokens  : int = 0
    past_key_values: Optional[tuple] = None
    generated_ids  : List[int] = field(default_factory=list)
    output_queue   : Optional[asyncio.Queue] = None
    created_at     : float = field(default_factory=time.perf_counter)
    first_token_at : Optional[float] = None
    finished_at    : Optional[float] = None

    @property
    def ttft(self):
        return (self.first_token_at - self.created_at) if self.first_token_at else None

    @property
    def total_time(self):
        return (self.finished_at - self.created_at) if self.finished_at else None


class ContinuousBatchingScheduler:

    def __init__(self, model, tokenizer, cache_manager, max_batch_size=MAX_BATCH_SIZE):
        self.model          = model
        self.tokenizer      = tokenizer
        self.cache_manager  = cache_manager
        self.max_batch_size = max_batch_size
        self.waiting_queue  : List[InferenceRequest] = []
        self.active_batch   : List[InferenceRequest] = []
        self.completed_requests : List[dict] = []
        self.total_steps    = 0

        # Pre-allocate reusable single-token tensor to avoid per-step allocation
        self._token_buf = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)

        print(f"Scheduler initialized | max_batch={max_batch_size} | device={DEVICE}")

    def submit(self, request: InferenceRequest):
        tokens = self.tokenizer(request.prompt, return_tensors="pt")
        request.input_ids     = tokens.input_ids.to(DEVICE)
        request.prompt_tokens = request.input_ids.shape[1]
        self.waiting_queue.append(request)

    def _admit_waiting_requests(self):
        """Prefill and admit requests one at a time until batch is full."""
        while (
            self.waiting_queue
            and len(self.active_batch) < self.max_batch_size
        ):
            req = self.waiting_queue[0]
            ok  = self.cache_manager.register_request(req.request_id, req.prompt_tokens)
            if not ok:
                break

            # Prefill: one forward pass for the full prompt
            with torch.no_grad():
                out = self.model(
                    input_ids       = req.input_ids,
                    past_key_values = None,
                    use_cache       = True,
                )

            req.past_key_values = out.past_key_values
            first_token         = out.logits[0, -1, :].argmax().item()
            req.generated_ids.append(first_token)
            req.first_token_at = time.perf_counter()
            self.cache_manager.on_token_generated(req.request_id)

            if req.output_queue:
                req.output_queue.put_nowait(
                    self.tokenizer.decode([first_token], skip_special_tokens=True)
                )

            self.waiting_queue.pop(0)
            self.active_batch.append(req)

    async def run_forever(self):
        """Background loop — runs continuously as an asyncio task."""
        print("[SCHEDULER] Loop started")
        while True:
            self._admit_waiting_requests()
            if self.active_batch:
                self._step_sync()
            await asyncio.sleep(0)

    def _step_sync(self):
        """
        Synchronous step: one decode token per active request.
        Kept synchronous to eliminate asyncio overhead in the hot path.
        """
        self.total_steps += 1
        completed = []

        for i, req in enumerate(self.active_batch):
            # Reuse buffer to avoid tensor allocation per step
            self._token_buf[0, 0] = req.generated_ids[-1]

            with torch.no_grad():
                out = self.model(
                    input_ids       = self._token_buf,
                    past_key_values = req.past_key_values,
                    use_cache       = True,
                )

            req.past_key_values = out.past_key_values
            next_token          = out.logits[0, -1, :].argmax().item()
            req.generated_ids.append(next_token)
            self.cache_manager.on_token_generated(req.request_id)

            if req.output_queue:
                req.output_queue.put_nowait(
                    self.tokenizer.decode([next_token], skip_special_tokens=True)
                )

            if (next_token == self.tokenizer.eos_token_id
                    or len(req.generated_ids) >= req.max_new_tokens):
                req.finished_at = time.perf_counter()
                completed.append(i)
                if req.output_queue:
                    req.output_queue.put_nowait(None)

        for i in reversed(completed):
            done = self.active_batch.pop(i)
            self.cache_manager.free_request(done.request_id)
            self.completed_requests.append({
                "request_id"      : done.request_id,
                "prompt_tokens"   : done.prompt_tokens,
                "generated_tokens": len(done.generated_ids),
                "ttft_sec"        : round(done.ttft or 0, 4),
                "total_time_sec"  : round(done.total_time or 0, 4),
                "tokens_per_sec"  : round(
                    len(done.generated_ids) / (done.total_time or 1), 2
                ),
            })
            print(f"  [DONE] {done.request_id}: "
                  f"{len(done.generated_ids)} tokens | "
                  f"TTFT={done.ttft:.3f}s | "
                  f"total={done.total_time:.3f}s")

    def batch_status(self):
        return {
            "active"   : len(self.active_batch),
            "waiting"  : len(self.waiting_queue),
            "completed": len(self.completed_requests),
            "step"     : self.total_steps,
            "cache"    : self.cache_manager.status(),
        }