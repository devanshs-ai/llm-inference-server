"""
Phase 3 — KV Cache Block Allocator (PagedAttention-style)

Core idea: instead of each request owning a contiguous, pre-sized slice of VRAM,
we divide GPU memory into fixed-size blocks and assign blocks to requests on demand.

Why this matters:
  Naive approach: reserve max_seq_len worth of memory per request upfront.
  If max_seq_len=512 but the request only uses 80 tokens, 432 tokens worth of
  VRAM sits empty but is LOCKED — no other request can use it.

  Block approach: allocate one 16-token block at a time, as the request needs it.
  When a request finishes at token 80, we free exactly the blocks it used.
  No gaps. No waste. Other requests immediately get those blocks.

This is the core innovation of vLLM (Kwon et al., 2023).
"""

"""
    this is the operating system of the gpus memory, while the phase two showed the memory
    growth problem, this script is pretty much the solution for it
    if the kv_cache is a hotel, this the smart front desk receptionist

    in phase two each user had one long snake of memory that kept on growing, this code gives 
    them instead of a long head, it gives them a vblock size of about 16 wherein they get another block
    when one block gets finished

"""

"""
External Fragmentation (The "Tightly Packed" Problem)
The Problem: In the "Naive" approach, you ask the GPU for one big, continuous slab of memory for a user.

User A wants 100 tokens. You find a spot for 100.

User B wants 100 tokens. You park them right next to User A.

The Crisis: User A finishes and leaves. You now have a hole of 100 tokens. But User C arrives and wants 150 tokens.

Even though you might have 200 total tokens of free space scattered around, you don't have 150 tokens in a single row. User C can't park, and you get an "Out of Memory" (OOM) error even though your GPU is half empty.

The Fix: By using Blocks, User C doesn't need a row of 150. They just need 10 blocks (16 tokens each) from anywhere in the "parking lot.


Internal Fragmentation (The "Over-Ordering" Problem)The Problem: Because resizing memory is slow, most 
naive systems "over-reserve" space.You think: "Most users won't go over 512 tokens."So, you reserve a 
512-token slab for every single person.The Waste: If a user only says "Hi" (2 tokens), the other 510 tokens 
($512 - 2$) are locked. They are "internal" to that user's reservation. No one else can touch them.
The Fix: With the KVCacheManager, we only give a user one block (16 tokens) to start.If they only use 
2 tokens, we only "waste" 14 tokens ($16 - 2$).Compared to wasting 510 tokens, 
this is a 97% reduction in waste.

Reservation Waste (The "What If" Problem)
There is a third secret waste called Reservation Waste.
Even if a user is going to use 512 tokens eventually, they aren't using them right now. In a naive system, 
you have to hold that empty space for them for the entire duration of the chat.

With PagedAttention, we stay "Lean." We keep that memory in the Free Pool so that if User B is having a sudden 
"burst" of words, they can use that space now, and we'll find more space for User A later when they actually 
need it.
"""

import torch
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# the constants
BLOCK_SIZE    = 16          # tokens per block — vLLM uses 16, this is standard
NUM_LAYERS    = 22          # TinyLlama transformer layers
NUM_KV_HEADS  = 4           # GQA: 4 KV heads (not 32 — we discovered this in Phase 2)
HEAD_DIM      = 64          # dimension per head
DTYPE         = torch.float16
BYTES_PER_EL  = 2           # fp16 = 2 bytes

# Memory per block = 2 (K+V) × layers × kv_heads × block_size × head_dim × bytes
BYTES_PER_BLOCK = (
    2 * NUM_LAYERS * NUM_KV_HEADS * BLOCK_SIZE * HEAD_DIM * BYTES_PER_EL
)

class BlockState(Enum):
    FREE     = "free"
    OCCUPIED = "occupied"

@dataclass
class Block:
    """A single fixed-size unit of KV cache memory."""
    block_id   : int
    state      : BlockState = BlockState.FREE
    request_id : Optional[str] = None   # which request owns this block
    last_used  : float = 0.0            # timestamp — used for LRU eviction

@dataclass
class RequestState:
    """Tracks everything about an in-flight request. it tracks which blocks belong to which guest, a request 
    might store blocks non continuously but they appear continuous to the model, it prevents fragmentation"""
    request_id      : str
    prompt_tokens   : int
    tokens_generated: int = 0
    block_ids       : List[int] = field(default_factory=list)  # blocks assigned to this request
    priority        : float = 0.0       # for priority-based eviction (higher = more valuable)
    created_at      : float = field(default_factory=time.perf_counter)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.tokens_generated

    @property
    def blocks_needed(self) -> int:
        """How many blocks does this request currently require?"""
        return (self.total_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE #the ceiling division formula is used 

    def update_priority(self):
        """
        Priority = fraction of generation complete.
        A request 90% done is more valuable to keep than one 5% done.
        Evicting a nearly-complete request wastes more work.
        """
        self.priority = self.tokens_generated / max(self.tokens_generated + 1, 1)


class KVCacheManager:
    """
    Manages a fixed pool of KV cache blocks on GPU.

    Two eviction policies implemented:
      1. LRU  — evict the block unused for longest time
      2. Priority — evict the request that has made least progress
    """

    def __init__(
        self,
        total_gpu_memory_gb : float = 4.0,   # how much VRAM to dedicate to KV cache
        eviction_policy     : str   = "lru",  # "lru" or "priority"
        device              : str   = "cuda",
    ):
        self.device          = device
        self.eviction_policy = eviction_policy

        # ── Calculate pool size from available memory ──────────────
        total_bytes  = int(total_gpu_memory_gb * 1e9)
        self.num_blocks = total_bytes // BYTES_PER_BLOCK

        print(f"KVCacheManager initialized:")
        print(f"  Dedicated memory  : {total_gpu_memory_gb:.1f} GB")
        print(f"  Bytes per block   : {BYTES_PER_BLOCK/1024:.1f} KB  "
              f"({BLOCK_SIZE} tokens × {NUM_LAYERS} layers × {NUM_KV_HEADS} heads × {HEAD_DIM} dim × 2 (K+V) × 2 bytes)")
        print(f"  Total blocks      : {self.num_blocks}")
        print(f"  Max tokens (total): {self.num_blocks * BLOCK_SIZE:,}")
        print(f"  Eviction policy   : {eviction_policy}")

        # ── Initialize the block pool ──────────────────────────────
        self.blocks: Dict[int, Block] = {
            i: Block(block_id=i) for i in range(self.num_blocks)
        }

        # ── Request registry ───────────────────────────────────────
        self.requests: Dict[str, RequestState] = {}

        # ── Stats ──────────────────────────────────────────────────
        self.stats = {
            "allocations"  : 0,
            "evictions"    : 0,
            "frees"        : 0,
            "peak_blocks"  : 0,
        }

    # ── Core allocation API ────────────────────────────────────────────────────

    def register_request(self, request_id: str, prompt_tokens: int) -> bool:
        """
        Register a new request. If the hotel is full, kick someone out 
        to make room for the new guest's initial bags (prompt).
        """
        state = RequestState(
            request_id   = request_id,
            prompt_tokens= prompt_tokens,
        )
        self.requests[request_id] = state # Add to registry first
        
        initial_blocks_needed = state.blocks_needed

        # --- THIS IS THE CHANGE ---
        allocated = self._allocate_blocks(request_id, initial_blocks_needed)
        
        if not allocated:
            # If we can't fit the new person, try to evict someone else
            evicted = self._evict(exclude_request_id=request_id)
            if evicted:
                self.stats["evictions"] += 1
                # Try allocating again now that a room is free
                allocated = self._allocate_blocks(request_id, initial_blocks_needed)

        if not allocated:
            # If we still can't fit (even after eviction), then we fail
            print(f"  [CACHE] Cannot register {request_id} — OOM even after eviction")
            del self.requests[request_id]
            return False
        # --------------------------

        print(f"  [CACHE] Registered {request_id}: "
              f"{prompt_tokens} prompt tokens → {initial_blocks_needed} blocks allocated")
        return True

    def on_token_generated(self, request_id: str) -> bool:
        """
        Called after each token is generated for a request.
        Allocates a new block if the request has crossed a block boundary.
        Returns False if allocation failed (triggers eviction).
        """
        if request_id not in self.requests:
            return False

        state = self.requests[request_id]
        state.tokens_generated += 1
        state.update_priority()

        current_blocks = len(state.block_ids)
        needed_blocks  = state.blocks_needed

        if needed_blocks > current_blocks:
            # Crossed into a new block — allocate it
            allocated = self._allocate_blocks(request_id, needed_blocks - current_blocks)
            if not allocated:
                # Try eviction first, then retry
                evicted = self._evict(exclude_request_id=request_id)
                if evicted:
                    self.stats["evictions"] += 1
                    allocated = self._allocate_blocks(request_id, needed_blocks - current_blocks)

                if not allocated:
                    print(f"  [CACHE] OOM: cannot grow {request_id} — eviction failed")
                    return False

        # Update last_used on all blocks for this request
        now = time.perf_counter()
        for bid in state.block_ids:
            self.blocks[bid].last_used = now

        occupied = self.occupied_block_count
        if occupied > self.stats["peak_blocks"]:
            self.stats["peak_blocks"] = occupied

        return True

    def free_request(self, request_id: str):
        """Release all blocks held by a completed request."""
        if request_id not in self.requests:
            return

        state = self.requests[request_id]
        for bid in state.block_ids:
            self.blocks[bid].state      = BlockState.FREE
            self.blocks[bid].request_id = None
            self.stats["frees"] += 1

        print(f"  [CACHE] Freed {request_id}: "
              f"released {len(state.block_ids)} blocks → "
              f"{self.free_block_count} free blocks now available")

        del self.requests[request_id]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _allocate_blocks(self, request_id: str, count: int) -> bool:
        """Find `count` free blocks and assign them to request_id."""
        free_blocks = [b for b in self.blocks.values() if b.state == BlockState.FREE]
        if len(free_blocks) < count:
            return False

        now = time.perf_counter()
        for block in free_blocks[:count]:
            block.state      = BlockState.OCCUPIED
            block.request_id = request_id
            block.last_used  = now
            self.stats["allocations"] += 1

            if request_id in self.requests:
                self.requests[request_id].block_ids.append(block.block_id)

        return True

    def _evict(self, exclude_request_id: str) -> bool:
        """
        Evict one request using the configured policy.
        exclude_request_id: never evict this request (it's the one needing memory).
        """
        candidates = {
            rid: state for rid, state in self.requests.items()
            if rid != exclude_request_id
        }
        if not candidates:
            return False

        if self.eviction_policy == "lru":
            # Evict the request whose blocks were least recently used
            victim_id = min(
                candidates,
                key=lambda rid: min(
                    self.blocks[bid].last_used
                    for bid in candidates[rid].block_ids
                )
            )
        elif self.eviction_policy == "priority":
            # Evict the request that has made least progress (lowest priority)
            victim_id = min(candidates, key=lambda rid: candidates[rid].priority)
        else:
            raise ValueError(f"Unknown eviction policy: {self.eviction_policy}")

        victim = candidates[victim_id]
        print(f"  [CACHE] Evicting {victim_id} "
              f"(policy={self.eviction_policy}, "
              f"tokens_generated={victim.tokens_generated}, "
              f"priority={victim.priority:.3f})")
        self.free_request(victim_id)
        return True

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def free_block_count(self) -> int:
        return sum(1 for b in self.blocks.values() if b.state == BlockState.FREE)

    @property
    def occupied_block_count(self) -> int:
        return self.num_blocks - self.free_block_count

    @property
    def utilization(self) -> float:
        return self.occupied_block_count / self.num_blocks

    def status(self) -> dict:
        return {
            "total_blocks"   : self.num_blocks,
            "free_blocks"    : self.free_block_count,
            "occupied_blocks": self.occupied_block_count,
            "utilization_pct": round(self.utilization * 100, 1),
            "active_requests": len(self.requests),
            "stats"          : self.stats,
        }