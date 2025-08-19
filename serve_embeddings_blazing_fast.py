# serve_embeddings_blazing_fast.py - Optimized for speed with batching and inference mode
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

# Disable parallelism bottlenecks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("🚀 Loading Qwen3-Embedding-8B for blazing fast inference...")
    start = time.time()
    
    # Load model on GPU
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
    model.eval()
    
    # Optimize for speed
    model.max_seq_length = 2048  # Reduced from 8192 for 2x speed on long docs
    
    # Enable TF32 for RTX 4060 (Ampere)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"📏 Max sequence length: {model.max_seq_length}")
    print(f"🎮 Device: {model.device}")
    
    # Warmup with inference mode
    with torch.inference_mode():
        _ = model.encode(["warmup"], batch_size=1, convert_to_numpy=True)
    
    print("⚡ Model ready for blazing fast inference!")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(
    title="TORI Blazing Fast Embedding Service",
    description="Optimized Qwen3-8B with batching and inference mode",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Extract and represent the semantic meaning"

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    # Format with instruction if needed
    formatted_texts = [f"Instruct: {request.instruction}\nQuery: {text}" 
                      for text in request.texts]
    
    # BLAZING FAST: Batch encoding with inference mode
    with torch.inference_mode():  # ❶ Disable autograd for speed
        embeddings = model.encode(
            formatted_texts,
            batch_size=len(formatted_texts),  # ❷ Process entire batch at once
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
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
        }
    
    return {
        "status": "healthy",
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": str(model.device) if model else "not_loaded",
        "max_seq_length": model.max_seq_length if model else "not_loaded",
        "cache_size": 0,
        "gpu_info": gpu_info
    }

if __name__ == "__main__":
    # Apply av.logging fix
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
    
    port = int(os.getenv("EMBED_PORT", 8080))
    
    uvicorn.run(app, host="0.0.0.0", port=port)
