# serve_embeddings_optimized_final.py - Best possible FP16 performance
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

# Optimize environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Enable TF32 for Ampere
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("⚡ Loading Qwen3-8B with maximum optimizations...")
    start = time.time()
    
    # Load model
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
    model.eval()
    
    # CRITICAL: Use FP16
    model = model.half()
    
    # CRITICAL: Reduce context length
    model.max_seq_length = 256  # Most texts don't need 512
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"📏 Using FP16 with max_seq_length=256")
    print(f"🎮 Device: {model.device}")
    
    # Warmup
    with torch.inference_mode():
        _ = model.encode(["warmup"], batch_size=1, show_progress_bar=False)
    
    print("🚀 Optimized for best FP16 performance!")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(
    title="TORI Optimized FP16 Service",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = ""  # Make optional

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    # Only add instruction if provided
    if request.instruction:
        formatted_texts = [f"Instruct: {request.instruction}\nQuery: {text}" 
                          for text in request.texts]
    else:
        formatted_texts = request.texts
    
    # CRITICAL: Proper batching with all optimizations
    with torch.inference_mode():
        embeddings = model.encode(
            formatted_texts,
            batch_size=min(32, len(formatted_texts)),
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
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
        }
    
    return {
        "status": "healthy",
        "model": "Qwen3-8B",
        "precision": "FP16",
        "max_seq_length": model.max_seq_length if model else None,
        "gpu_info": gpu_info
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
