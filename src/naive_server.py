import time
import asyncio
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Configuration ---
# Use the absolute path to your local model folder
MODEL_ID   = "C:/Users/Devansh/holeeshit/llm-inference-server/models/TinyLlama-1.1B-Chat"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 100

app = FastAPI(title="Phase 1: Naive LLM Server")

print(f"Loading model on {DEVICE}...")

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16, # model is loaded in 16 bit format to cut the memory usage by 50 %
    device_map=DEVICE # cuda or cpu
).eval() # get rid of dropout and reduce uncertainty

print(f"Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# --- Request / Response Schemas ---
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = MAX_NEW_TOKENS

class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    total_time_sec: float
    tokens_per_sec: float

# --- The Naive Endpoint ---
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Naive implementation: processes one request at a time (synchronous/blocking).
    """
    t_start = time.perf_counter() #perf_counter is used for benchmarking tasks as .time() interferes by syncing with local clock

    # 1. Tokenization
    inputs = tokenizer(request.prompt, return_tensors="pt").to(DEVICE)
    prompt_len = inputs["input_ids"].shape[1]

    # 2. Generation
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=request.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    t_end = time.perf_counter()

    # 3. Decode
    new_ids = output_ids[0][prompt_len:]
    generated_text = tokenizer.decode(new_ids, skip_special_tokens=True)
    generated_len  = len(new_ids)
    total_time     = t_end - t_start

    # 4. Return structured response
    return GenerateResponse(
        generated_text  = generated_text,
        prompt_tokens   = prompt_len,
        generated_tokens= generated_len,
        total_time_sec  = round(total_time, 3),
        tokens_per_sec  = round(generated_len / total_time, 2)
    )

@app.get("/health")
async def health():
    free, total = torch.cuda.mem_get_info()
    return {
        "status": "ok",
        "vram_free_gb": round(free / 1e9, 2),
        "vram_used_gb": round((total - free) / 1e9, 2),
    }