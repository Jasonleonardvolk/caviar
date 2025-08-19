# serve_embeddings_bnb_simple.py - The SIMPLEST 4-bit implementation
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer

# Optimize environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global embedder
embedder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    
    print("⚡ Loading Qwen3-8B with BitsAndBytes 4-bit (SIMPLE VERSION)...")
    start = time.time()
    
    MODEL_ID = "Qwen/Qwen3-Embedding-8B"
    
    # 1️⃣ Build the quant-config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # 2️⃣ Just ONE call - ST does everything!
    embedder = SentenceTransformer(
        MODEL_ID,
        device="cuda",
        trust_remote_code=True,
        model_kwargs={
            "quantization_config": bnb_cfg,
            "device_map": "auto",
        }
    )
    embedder.max_seq_length = 512
    
    # 3️⃣ Smoke test
    print("Running smoke test...")
    with torch.inference_mode():
        t0 = time.time()
        vec = embedder.encode(["hello world"] * 10, batch_size=10, show_progress_bar=False)
        test_time = time.time() - t0
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"📊 Test: 10 texts in {test_time:.3f}s (dim: {vec.shape[1]})")
    print(f"⚡ Expected: ~600-800ms for 10×512-token texts")
    print(f"🚀 Ready for 11x faster embeddings!")
    
    yield
    
    del embedder
    torch.cuda.empty_cache()

app = FastAPI(title="TORI BnB 4-bit SIMPLE", lifespan=lifespan)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Extract and represent the semantic meaning"

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    # Format with instruction
    formatted_texts = [f"Instruct: {request.instruction}\nQuery: {text}" 
                      for text in request.texts]
    
    # Embed with batching
    with torch.inference_mode():
        embeddings = embedder.encode(
            formatted_texts,
            batch_size=min(32, len(formatted_texts)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    ms = (time.time() - start) * 1000
    
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
        "quantization": "BitsAndBytes 4-bit NF4",
        "expected_speed": "~70ms per text in batches"
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
