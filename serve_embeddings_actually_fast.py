# serve_embeddings_actually_fast.py - Working optimized version
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

# Critical optimizations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable mixed precision on RTX 4060
torch.set_float32_matmul_precision('high')

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("⚡ Loading Qwen3-8B with WORKING optimizations...")
    start = time.time()
    
    # Load model
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
    model.eval()
    
    # CRITICAL FIX 1: Convert to FP16
    print("Converting to FP16...")
    model = model.half()
    
    # CRITICAL FIX 2: Reduce sequence length
    model.max_seq_length = 256  # Most texts are <256 tokens
    
    # CRITICAL FIX 3: Disable gradient checkpointing if enabled
    for module in model.modules():
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = False
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"📏 Max sequence length: {model.max_seq_length}")
    print(f"🎮 Using FP16 on: {model.device}")
    
    # Warmup with proper batching
    print("Warming up...")
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            _ = model.encode(
                ["warmup text"],
                batch_size=1,
                convert_to_numpy=True,
                show_progress_bar=False
            )
    
    print("🚀 Ready! Expecting ~400ms for 10 texts")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(
    title="TORI Actually Fast Service",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = ""

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    # Skip instruction formatting if empty
    if request.instruction:
        texts = [f"Instruct: {request.instruction}\nQuery: {text}" for text in request.texts]
    else:
        texts = request.texts
    
    # Use autocast for additional speed
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            embeddings = model.encode(
                texts,
                batch_size=min(32, len(texts)),
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
    
    elapsed = time.time() - start
    ms = elapsed * 1000
    
    print(f"⚡ {len(request.texts)} texts in {ms:.1f}ms ({ms/len(request.texts):.1f}ms per text)")
    
    return {
        "embeddings": embeddings.tolist(),
        "processing_time_ms": ms,
        "texts_processed": len(request.texts),
        "cache_hits": 0,
        "cache_misses": len(request.texts)
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "Qwen3-8B",
        "precision": "FP16",
        "max_seq_length": model.max_seq_length if model else None,
        "device": str(model.device) if model else None
    }

if __name__ == "__main__":
    import importlib, types
    try:
        av = importlib.import_module("av")
        if not hasattr(av, "logging"):
            av.logging = types.SimpleNamespace(
                ERROR=0, WARNING=1, INFO=2, DEBUG=3,
                set_level=lambda *_, **__: None,
            )
    except:
        pass
    
    uvicorn.run(app, host="0.0.0.0", port=8080)
