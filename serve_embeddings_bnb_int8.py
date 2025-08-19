# serve_embeddings_bnb_int8.py - Using bitsandbytes INT8
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("⚡ Loading Qwen3-8B with INT8 quantization...")
    start = time.time()
    
    try:
        import bitsandbytes as bnb
        
        # Load with INT8
        model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-8B",
            device="cuda",
            model_kwargs={
                "load_in_8bit": True,
                "device_map": "auto",
                "bnb_4bit_compute_dtype": torch.float16
            }
        )
        
        print("✅ Loaded with INT8 quantization!")
        
    except ImportError:
        print("⚠️ bitsandbytes not installed, using FP16")
        print("   Install with: pip install bitsandbytes")
        
        # Fallback to FP16
        model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
        model = model.half()
    
    model.eval()
    model.max_seq_length = 256
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"📏 Max sequence length: {model.max_seq_length}")
    
    # Warmup
    with torch.inference_mode():
        _ = model.encode(["warmup"], batch_size=1, show_progress_bar=False)
    
    print("🚀 Ready!")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = ""

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    texts = request.texts
    if request.instruction:
        texts = [f"Instruct: {request.instruction}\nQuery: {text}" for text in texts]
    
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=min(32, len(texts)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    ms = (time.time() - start) * 1000
    
    print(f"⚡ {len(texts)} texts in {ms:.1f}ms ({ms/len(texts):.1f}ms per text)")
    
    return {
        "embeddings": embeddings.tolist(),
        "processing_time_ms": ms,
        "texts_processed": len(texts),
        "cache_hits": 0,
        "cache_misses": len(texts)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "Qwen3-8B-INT8"}

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
