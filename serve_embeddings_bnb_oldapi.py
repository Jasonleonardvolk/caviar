# serve_embeddings_bnb_oldapi.py - Works with OLD sentence-transformers
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

# Global model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("⚡ Loading Qwen3-8B with BitsAndBytes 4-bit (OLD API)...")
    start = time.time()
    
    # BitsAndBytes config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    # Load with SentenceTransformer directly (old API)
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        device="cuda",
        model_kwargs={
            "quantization_config": bnb_cfg,
            "device_map": "auto",
            "trust_remote_code": True
        }
    )
    model.max_seq_length = 512
    
    # Test
    print("Running test...")
    with torch.inference_mode():
        t0 = time.time()
        vecs = model.encode(["test"]*10, batch_size=10, show_progress_bar=False)
        test_time = time.time() - t0
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"📊 Test: 10 texts in {test_time:.3f}s")
    print(f"🚀 Ready!")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(title="TORI BnB Old API", lifespan=lifespan)

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
        "texts_processed": len(texts)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "Qwen3-8B-4bit"}

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
