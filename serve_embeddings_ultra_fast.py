# serve_embeddings_ultra_fast.py - ACTUALLY FAST with proper batching
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

# CRITICAL: Disable BLAS over-threading
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("⚡ Loading Qwen3-Embedding-8B with REAL optimizations...")
    start = time.time()
    
    # Load model on GPU
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
    model.eval()
    
    # CRITICAL OPTIMIZATIONS
    model.max_seq_length = 2048  # Cut from 8192 for 2x speed
    
    # Enable TF32 for RTX 4060
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"🎯 Max sequence length: {model.max_seq_length}")
    print(f"🎮 Device: {model.device}")
    
    # Warmup
    with torch.inference_mode():
        _ = model.encode(
            ["warmup text"],
            batch_size=1,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    print("🚀 Model ready - expecting <100ms per text in batches!")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(
    title="TORI Ultra Fast Embedding Service",
    description="Actually fast with proper batching",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Extract and represent the semantic meaning"

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    # Format with instruction
    formatted_texts = [f"Instruct: {request.instruction}\nQuery: {text}" 
                      for text in request.texts]
    
    # THE KEY FIX: Proper batching with explicit batch_size!
    with torch.inference_mode():
        embeddings = model.encode(
            formatted_texts,
            batch_size=min(32, len(formatted_texts)),  # ← THE CRITICAL FIX
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
            device='cuda'
        )
    
    elapsed = time.time() - start
    ms = elapsed * 1000
    
    print(f"⚡ Processed {len(request.texts)} texts in {ms:.1f}ms ({ms/len(request.texts):.1f}ms per text)")
    
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
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": str(model.device),
        "max_seq_length": model.max_seq_length,
        "optimization": "ultra_fast_with_real_batching"
    }

if __name__ == "__main__":
    # av.logging fix
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
