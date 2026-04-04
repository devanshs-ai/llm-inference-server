"""
Phase 2 — Manual Token Control.

We replace model.generate() with a hand-written loop that calls model.forward()
one step at a time. After this file, you will have seen and touched:
  - The raw logits tensor (model's "thoughts" before picking a word)
  - The past_key_values structure (the KV cache we manage in Phase 3)
  - The exact memory cost per token per layer
"""

'''
    understanding the kv cache, now you are manually catching the past_key_values output
    feeding it back in, 
    engineering problem : we are watching the cache size (mb) to grow. This proves that 
    a longer conversation means it takes more memory hence increasing the load on the gpu

    distinguish prefill and decode 
    prefill, the model processes the 10 word prompt all at once, fast and parallel
    decode, however the model generates the words one by one by one by one, which is slow 
    by writing this manual loop we see that prefill happens only once but the decode happens 100 times

    preparation for continuous batching, model.generate() is a blocking function, if we are generating
    for user a, we cannot generate for user b until user a is done, which proves to be inefficient,
    but since now we control the flow, we can eventually write code which says okay i finished token 1 
    for user a, but before i move forward let me just check if user b has a new request, if yes
    ill do one token for both of them in the next loop achieving better throughput

    fragmentation, seqlen is the one thing always changing, if we keep on increasing memory
    by the time for everyone, they will bump into each other, this is memory fragmentation
    hence we require, paged attention, breaking memory into fixed blocks
'''

from numpy import float16
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID   = "C:/Users/Devansh/holeeshit/llm-inference-server/venv/models/TinyLlama-1.1B-Chat"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

#load the model

print(f"Loading model {MODEL_ID} on {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype = torch.float16,
    device_map = DEVICE
).eval()

def print_kv_cache_stats(past_key_values, step: int):
    """
    Inspect the KV cache structure so you can see exactly what you'll manage.
    past_key_values is a tuple of tuples:
      - Outer tuple: one entry per transformer layer (TinyLlama has 22 layers)
      - Inner tuple: (key_tensor, value_tensor) for that layer
      - Each tensor shape: [batch_size, num_heads, seq_len_so_far, head_dim]
    """
    num_layers = len(past_key_values)
    k_tensor = past_key_values[0][0]
    batch_size, num_heads, seq_len, head_dim = k_tensor.shape
    
    #2 for key and value both
    bytes_per_token = 2 * num_layers * batch_size * head_dim * 2 # 2 bytes for float16 bits
    total_bytes = bytes_per_token * seq_len

    if step % 10 == 0 or step < 3:
        print(f"  Step {step:3d} | seq_len={seq_len:4d} | "
              f"KV shape: {list(k_tensor.shape)} | "
              f"Cache size: {total_bytes/1e6:.2f} MB")

def greedy_sample(logits: torch.Tensor) -> torch.Tensor :
    """
    Greedy decoding: pick the token with highest probability.
    logits shape: [batch_size, vocab_size]
    This is what model.generate() does internally with do_sample=False.
    """
    return logits.argmax(dim=-1)

def manual_generate(prompt:str, max_new_tokens: int=80) -> str:
    """
    The hand-written replacement for model.generate().
    Every line here is something model.generate() was hiding from us.
    """
    print(f"\nPrompt: '{prompt[:60]}...'")
    print(f"{'─'*60}")
    
    #tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    print(f"Prompt tokenized: {input_ids.shape[1]} tokens")

    #prefill pass, process entire prompt in one pass, Output: logits for the next token + initial KV cache.
    
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            past_key_values = None,     # no cache yet — first pass
            use_cache = True    # tells model to return past_key_values
        )

    t_prefill_end = time.perf_counter()

    # Extract what the model returned
    logits          = outputs.logits[:, -1, :]   # logits for position AFTER the prompt
    past_key_values = outputs.past_key_values    # the KV cache — this grows each step

    print(f"Prefill done in {(t_prefill_end-t_prefill_start)*1000:.1f}ms")
    print(f"Logits shape: {list(logits.shape)}  (batch=1, vocab_size={logits.shape[-1]})")
    print(f"\nKV Cache structure:")
    print(f"  Layers: {len(past_key_values)}")
    print(f"  Per layer: (keys={list(past_key_values[0][0].shape)}, "
          f"values={list(past_key_values[0][1].shape)})")
    print(f"\nToken generation loop:")

    # Step 3: Sample the first new token
    next_token_id = greedy_sample(logits)       # shape: [1]
    generated_ids = [next_token_id.item()]

    ''' 
        this is the engine of the code, basically gpt, claude, llama all work using this under the hood
        sampling the first token, before the loop starts we take the prompt from the prefill stage
        we pick the most probable word and put in a list, this probable is the very first word in the 
        answer produced by the model

        an llm is like a person who can think only one word at a time
        it cannot plan a whole sentence, it looks at a word, to decide the next word, it is called 
        autoregressive generation

        the forward pass -->  this is the most important part 
        in the input instead of sending the entire prompt again, we are only sending the single last
        word generated, making the math lighter

        we are sending the model its notes (KV cache)

        updating the memory 
        the logic gives us back its guess for the next word and the updated notes block with 
        the new thoughts about the next word added 

        the eos token is the llms signal of end of sentence to stop the generation and break the loop

    '''

    t_decode_start = time.perf_counter()

    for step in range(1, max_new_tokens):
        print_kv_cache_stats(past_key_values, step)

        with torch.no_grad():
            outputs = model(
                input_ids = next_token_id.unsqueeze(0),
                past_key_values = past_key_values,
                use_cache = True
            )

        logits          = outputs.logits[:, -1, :]
        past_key_values = outputs.past_key_values   # now one token longer

        next_token_id = greedy_sample(logits)
        generated_ids.append(next_token_id.item())

        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"  EOS token at step {step} — generation complete")
            break

    t_decode_end = time.perf_counter()

    num_layers   = len(past_key_values)
    k            = past_key_values[0][0]
    batch, heads, seq_len, head_dim = k.shape
    final_bytes  = 2 * num_layers * batch * heads * head_dim * 2 * seq_len
    decode_time  = t_decode_end - t_decode_start
    tokens_gen   = len(generated_ids)

    print(f"\n{'─'*60}")
    print(f"Generation complete")
    print(f"  Tokens generated : {tokens_gen}")
    print(f"  Decode time      : {decode_time:.3f}s")
    print(f"  Tokens/sec       : {tokens_gen/decode_time:.1f}")
    print(f"  Final KV cache   : {final_bytes/1e6:.2f} MB for {seq_len} tokens")
    print(f"  Cost per token   : {final_bytes/seq_len/1e3:.2f} KB")
    print(f"  Key insight      : at batch=8, this would be {final_bytes*8/1e6:.1f} MB")
    print(f"                     each user gets their own cache — no sharing possible")
    print(f"                     THIS is the fragmentation problem Phase 3 solves")

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return generated_text

# Running it
if __name__ == "__main__":
    result = manual_generate(
        prompt="Explain what a transformer neural network is, in simple terms:",
        max_new_tokens=60,
    )
    print(f"\n{'='*60}")
    print("GENERATED TEXT:")
    print(result)
    print('='*60)

    # ── The key experiment: what does the cache actually look like? ──
    print("\n=== ANATOMY OF past_key_values ===")
    print("""
    past_key_values is a tuple of length = num_transformer_layers (22 for TinyLlama)
    Each element is a tuple: (key_tensor, value_tensor)
    Each tensor has shape: [batch_size, num_heads, sequence_length, head_dim]

    For TinyLlama after generating 80 tokens on a single request:
      batch_size = 1
      num_heads  = 32
      seq_len    = ~90 (80 new + ~10 prompt tokens)
      head_dim   = 64

    Memory per request = 2 (K+V) × 22 (layers) × 1 × 32 × 90 × 64 × 2 bytes
                       ≈ 16 MB per request

    Scale to 8 concurrent requests = 128 MB just for KV caches
    Scale to 50 concurrent requests = 800 MB — nearly 10% of your entire VRAM
    gone just on KV state, with huge gaps because sequences finish at different times.

    The block allocator in Phase 3 fixes this by:
      - Pre-allocating a fixed pool of 16-token blocks
      - Assigning blocks to requests on demand
      - Reclaiming blocks the moment a request finishes
      - No gaps, no waste, no per-request pre-allocation
    """)




