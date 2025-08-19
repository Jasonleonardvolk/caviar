# serve_embeddings_simple.py - Minimal service without instruction formatting
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
from contextlib import asynccontextmanager

# Disable parallelism
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading model...")
    start = time.time()
    
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
    model.eval()
    model.max_seq_length = 512  # Even shorter for testing
    
    print(f"Model loaded in {time.time() - start:.2f}s")
    
    # Warmup
    with torch.inference_mode():
        _ = model.encode(["test"], convert_to_numpy=True)
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class Request(BaseModel):
    texts: List[str]

@app.post("/embed")
async def embed(request: Request):
    start = time.time()
    
    # Direct encoding without instruction formatting
    with torch.inference_mode():
        embeddings = model.encode(
            request.texts,
            batch_size=len(request.texts),
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    ms = (time.time() - start) * 1000
    
    return {
        "embeddings": embeddings.tolist(),
        "processing_time_ms": ms,
        "texts_processed": len(request.texts)
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "device": str(model.device)}

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
