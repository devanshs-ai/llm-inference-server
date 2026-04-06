"""
Tests for KVCacheManager — all four tests should show eviction working correctly.
"""
import time
from src.kv_cache_manager import KVCacheManager, BYTES_PER_BLOCK, BLOCK_SIZE

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_basic_lifecycle():
    separator("TEST 1: Basic lifecycle — alloc, grow, free")

    mgr = KVCacheManager(total_gpu_memory_gb=0.1, eviction_policy="lru")
    initial_free = mgr.free_block_count

    ok = mgr.register_request("req_A", prompt_tokens=15)
    assert ok, "Registration failed"
    assert "req_A" in mgr.requests, "Request not in registry"
    assert len(mgr.requests["req_A"].block_ids) > 0, "block_ids empty after registration"

    print(f"\nAfter registration:")
    print(f"  block_ids populated: {mgr.requests['req_A'].block_ids}")

    # Generate 40 tokens — should cross 3 block boundaries (15+40=55 tokens = 4 blocks)
    for i in range(40):
        ok = mgr.on_token_generated("req_A")
        assert ok, f"Failed at token {i}"

    state = mgr.requests["req_A"]
    print(f"\nAfter 40 tokens generated:")
    print(f"  Total tokens    : {state.total_tokens}  (15 prompt + 40 generated)")
    print(f"  Blocks used     : {len(state.block_ids)}")
    print(f"  Expected blocks : {state.blocks_needed}")
    print(f"  Block IDs       : {state.block_ids}")
    assert len(state.block_ids) == state.blocks_needed, "Block count mismatch"

    mgr.free_request("req_A")
    assert mgr.free_block_count == initial_free, "Memory leak — not all blocks returned"
    print(f"\n  Memory leak check: PASSED (all {initial_free} blocks returned)")

def test_concurrent_requests():
    separator("TEST 2: Concurrent requests — uneven generation + early free")

    mgr = KVCacheManager(total_gpu_memory_gb=0.1, eviction_policy="lru")

    for rid in ["req_1", "req_2", "req_3"]:
        ok = mgr.register_request(rid, prompt_tokens=10)
        assert ok
        assert len(mgr.requests[rid].block_ids) > 0, f"{rid} block_ids empty"

    # Uneven generation: req_2 goes much further
    for _ in range(5):  mgr.on_token_generated("req_1")
    for _ in range(40): mgr.on_token_generated("req_2")  # crosses 3 block boundaries
    for _ in range(3):  mgr.on_token_generated("req_3")

    print(f"\nAfter uneven generation:")
    for rid in ["req_1", "req_2", "req_3"]:
        if rid in mgr.requests:
            st = mgr.requests[rid]
            print(f"  {rid}: {st.tokens_generated:2d} tokens | "
                  f"{len(st.block_ids)} blocks | "
                  f"block_ids={st.block_ids} | "
                  f"priority={st.priority:.3f}")

    free_before = mgr.free_block_count
    mgr.free_request("req_2")
    free_after  = mgr.free_block_count
    released    = free_after - free_before

    print(f"\n  req_2 freed: {released} blocks immediately returned")
    assert released > 1, "req_2 should have held multiple blocks"
    print(f"  Fragmentation test: PASSED")

def test_lru_eviction():
    separator("TEST 3: LRU eviction under genuine memory pressure")

    # 1. Pool exactly large enough for 2 blocks total
    bytes_for_2_blocks = BYTES_PER_BLOCK * 2
    gb_for_2_blocks = bytes_for_2_blocks / 1e9
    mgr = KVCacheManager(total_gpu_memory_gb=gb_for_2_blocks, eviction_policy="lru")

    print(f"\nPool size: {mgr.num_blocks} blocks (The 'Hotel' only has 2 rooms)")
    assert mgr.num_blocks == 2

    # 2. Register 'oldest_req' - Takes Block 0
    mgr.register_request("oldest_req", prompt_tokens=5)
    mgr.on_token_generated("oldest_req") # Give it a timestamp
    time.sleep(0.05) # Ensure it is definitively the 'oldest'

    # 3. Register 'middle_req' - Takes Block 1
    # NOW THE POOL IS 100% FULL (2/2)
    mgr.register_request("middle_req", prompt_tokens=5)
    print(f"  Pool is now FULL: {mgr.occupied_block_count}/{mgr.num_blocks} blocks used")

    # 4. Register 'newcomer_req' - This MUST trigger an eviction
    print(f"  Registering 'newcomer_req' — should trigger LRU eviction of 'oldest_req'...")
    ok = mgr.register_request("newcomer_req", prompt_tokens=5)

    # --- Verification ---
    print(f"\n  Evictions triggered : {mgr.stats['evictions']}")
    print(f"  oldest_req still alive : {'oldest_req' in mgr.requests}")
    print(f"  middle_req still alive : {'middle_req' in mgr.requests}")
    print(f"  newcomer_req alive     : {'newcomer_req' in mgr.requests}")

    assert ok, "Newcomer should have been registered after eviction"
    assert mgr.stats["evictions"] >= 1, "Bouncer should have kicked someone out!"
    assert "oldest_req" not in mgr.requests, "The oldest guest should be gone."
    assert "newcomer_req" in mgr.requests, "The newcomer should have their room."
    
    print(f"  LRU eviction test: PASSED")

def test_priority_eviction():
    separator("TEST 4: Priority eviction — protect nearly-done requests")

    # 1. We need 4 blocks total:
    # late_req needs 3 blocks (33 tokens)
    # early_req needs 1 block (7 tokens)
    # Total = 4 blocks.
    bytes_for_4_blocks = BYTES_PER_BLOCK * 4
    gb_for_4_blocks = bytes_for_4_blocks / 1e9
    mgr = KVCacheManager(total_gpu_memory_gb=gb_for_4_blocks, eviction_policy="priority")

    print(f"\nPool: {mgr.num_blocks} blocks (The 'Hotel' has 4 rooms)")
    assert mgr.num_blocks == 4

    # 2. Setup early_req (Low Progress/Priority)
    # 5 prompt + 2 gen = 7 tokens (1 block)
    mgr.register_request("early_req", prompt_tokens=5)
    for _ in range(2):
        mgr.on_token_generated("early_req")

    # 3. Setup late_req (High Progress/Priority)
    # 5 prompt + 28 gen = 33 tokens (3 blocks)
    mgr.register_request("late_req", prompt_tokens=5)
    for _ in range(28):
        mgr.on_token_generated("late_req")

    # Pool should now be 100% full (1 + 3 = 4 blocks)
    print(f"\n  Pool state before pressure: {mgr.occupied_block_count}/{mgr.num_blocks} blocks used")
    for rid in ["early_req", "late_req"]:
        st = mgr.requests[rid]
        print(f"  {rid}: {st.total_tokens} total tokens | {len(st.block_ids)} blocks | priority={st.priority:.3f}")

    # 4. Register pressure_req — Pool is full. 
    # Bouncer looks at priorities: 
    # early_req (0.667) vs late_req (0.966). 
    # It SHOULD kick out early_req to free up 1 block for pressure_req.
    print(f"\n  Registering pressure_req — pool full, must evict...")
    ok = mgr.register_request("pressure_req", prompt_tokens=5)

    print(f"\n  Evictions triggered : {mgr.stats['evictions']}")
    print(f"  early_req alive     : {'early_req' in mgr.requests}")
    print(f"  late_req alive      : {'late_req' in mgr.requests}")
    print(f"  pressure_req alive  : {'pressure_req' in mgr.requests}")

    # Final Asserts
    assert ok, "pressure_req should have succeeded after eviction"
    assert mgr.stats["evictions"] >= 1, "Priority eviction should have triggered"
    assert "early_req" not in mgr.requests, "early_req (low priority) should have been sacrificed"
    assert "late_req" in mgr.requests, "late_req (high priority) should have survived"
    
    print(f"  Priority eviction test: PASSED")
def test_block_ids_integrity():
    separator("TEST 5: block_ids integrity — no phantom blocks")

    mgr = KVCacheManager(total_gpu_memory_gb=0.1, eviction_policy="lru")

    mgr.register_request("req_X", prompt_tokens=8)

    # Generate enough tokens to span 4 blocks
    for _ in range(60):
        mgr.on_token_generated("req_X")

    state = mgr.requests["req_X"]

    # Every block_id in state.block_ids must be OCCUPIED and owned by req_X
    for bid in state.block_ids:
        block = mgr.blocks[bid]
        assert block.state.value == "occupied", f"Block {bid} should be OCCUPIED"
        assert block.request_id == "req_X",     f"Block {bid} owned by wrong request"

    # No block NOT in state.block_ids should be owned by req_X
    phantom = [
        bid for bid, b in mgr.blocks.items()
        if b.request_id == "req_X" and bid not in state.block_ids
    ]
    assert len(phantom) == 0, f"Phantom blocks found: {phantom}"

    print(f"\n  Block ID integrity across {len(state.block_ids)} blocks: PASSED")
    print(f"  No phantom allocations: PASSED")

    mgr.free_request("req_X")
    orphans = [bid for bid, b in mgr.blocks.items() if b.request_id == "req_X"]
    assert len(orphans) == 0, f"Orphaned blocks after free: {orphans}"
    print(f"  No orphaned blocks after free: PASSED")

if __name__ == "__main__":
    test_basic_lifecycle()
    test_concurrent_requests()
    test_lru_eviction()
    test_priority_eviction()
    test_block_ids_integrity()

    print(f"\n{'='*60}")
    print(f"  All 5 tests passed.")
    print(f"{'='*60}")