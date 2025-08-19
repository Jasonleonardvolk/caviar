# serve_embeddings_fast.py - Optimized embedding service
import os
import asyncio
import logging
import time
from typing import List
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force CUDA if available
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available - will run on CPU (slow!)")

# Global model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup"""
    global model
    
    logger.info("Loading Qwen3-Embedding-8B model...")
    start_time = time.time()
    
    # Force GPU and optimal settings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        device=device
    )
    
    # Move model to GPU and set to eval mode
    model.eval()
    if device == "cuda":
        model = model.half()  # Use FP16 for faster inference
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s on device: {model.device}")
    
    # Warm up the model
    logger.info("Warming up model...")
    with torch.no_grad():
        _ = model.encode(["Warmup text"], convert_to_tensor=False)
    logger.info("Model ready!")
    
    yield
    
    if model:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

app = FastAPI(
    title="TORI Fast Embedding Service",
    version="1.0.0",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    processing_time_ms: float

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embeddings quickly"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        start_time = time.time()
        
        # Process without instruction prefix for speed
        with torch.no_grad():
            embeddings = model.encode(
                request.texts,
                normalize_embeddings=True,
                convert_to_tensor=False,
                batch_size=32,
                show_progress_bar=False
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Processed {len(request.texts)} texts in {processing_time:.2f}ms")
        
        return EmbedResponse(
            embeddings=embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": str(model.device) if model else "not_loaded",
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", 8080))
    
    # Apply av.logging fix
    import importlib, types
    try:
        av = importlib.import_module("av")
        if not hasattr(av, "logging"):
            av.logging = types.SimpleNamespace(
                ERROR=0, WARNING=1, INFO=2, DEBUG=3,
                set_level=lambda *_, **__: None,
            )
    except ModuleNotFoundError:
        pass
    
    uvicorn.run(
        "serve_embeddings_fast:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
        access_log=True
    )
