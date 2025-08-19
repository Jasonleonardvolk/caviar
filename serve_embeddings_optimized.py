# serve_embeddings_optimized.py - Optimized with batching and inference mode
import os
import asyncio
import logging
import time
from typing import List, Dict, Any
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import hashlib
import diskcache as dc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable parallelism bottlenecks
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model and cache
model = None
cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and cache on startup"""
    global model, cache
    
    logger.info("Loading Qwen3-Embedding-8B model...")
    start_time = time.time()
    
    # Load model on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device)
    
    # Optimize for inference
    model.eval()
    model.max_seq_length = 2048  # Reduce from 8192 for faster processing
    
    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s on device: {model.device}")
    logger.info(f"Max sequence length set to: {model.max_seq_length}")
    
    # Initialize cache
    cache_dir = "./emb_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache = dc.Cache(cache_dir)
    
    # Warm up
    logger.info("Warming up model...")
    with torch.inference_mode():
        _ = model.encode(["Warmup"], batch_size=1, show_progress_bar=False)
    
    logger.info("Model ready for optimized inference!")
    
    yield
    
    # Cleanup
    if model:
        del model
        torch.cuda.empty_cache()
    if cache:
        cache.close()

app = FastAPI(
    title="TORI Optimized Embedding Service",
    description="Batched embedding service with Qwen3-Embedding-8B",
    version="2.0.0",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Extract and represent the semantic meaning"
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    cache_hits: int
    cache_misses: int
    processing_time_ms: float
    batch_size: int

def format_text_with_instruction(text: str, instruction: str) -> str:
    """Format text with instruction prefix for Qwen3"""
    return f"Instruct: {instruction}\nQuery: {text}"

async def embed_cached_batch(texts: List[str], instruction: str) -> tuple:
    """Cached embedding with proper batching"""
    start_time = time.time()
    
    uncached_indices = []
    uncached_texts = []
    results = [None] * len(texts)
    cache_hits = 0
    cache_misses = 0
    
    # Check cache
    for i, text in enumerate(texts):
        cache_key = hashlib.sha256(f"{instruction}::{text}".encode()).hexdigest()
        
        if cache_key in cache:
            results[i] = cache[cache_key]
            cache_hits += 1
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)
            cache_misses += 1
    
    # Process uncached texts in a single batch
    if uncached_texts:
        formatted_texts = [format_text_with_instruction(text, instruction) for text in uncached_texts]
        
        # Single batched inference call
        with torch.inference_mode():
            embeddings = model.encode(
                formatted_texts,
                batch_size=len(formatted_texts),  # Process all at once
                normalize_embeddings=True,
                convert_to_numpy=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                show_progress_bar=False
            )
        
        # Cache results
        for idx, emb, text in zip(uncached_indices, embeddings, uncached_texts):
            cache_key = hashlib.sha256(f"{instruction}::{text}".encode()).hexdigest()
            emb_list = emb.tolist()
            cache[cache_key] = emb_list
            results[idx] = emb_list
    
    processing_time = (time.time() - start_time) * 1000
    return results, cache_hits, cache_misses, processing_time

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embeddings with batching"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > 1000:
            raise HTTPException(status_code=400, detail="Too many texts (max 1000)")
        
        embeddings, cache_hits, cache_misses, processing_time = await embed_cached_batch(
            request.texts, 
            request.instruction
        )
        
        logger.info(f"Processed {len(request.texts)} texts in {processing_time:.2f}ms (batch mode)")
        
        return EmbedResponse(
            embeddings=embeddings,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            processing_time_ms=processing_time,
            batch_size=len(request.texts)
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
            "memory_reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"
        }
    
    return {
        "status": "healthy",
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": str(model.device) if model else "not_loaded",
        "max_seq_length": model.max_seq_length if model else "not_loaded",
        "cache_size": len(cache) if cache else 0,
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
    except ModuleNotFoundError:
        pass
    
    port = int(os.getenv("EMBED_PORT", 8080))
    
    uvicorn.run(
        "serve_embeddings_optimized:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
        access_log=True
    )
