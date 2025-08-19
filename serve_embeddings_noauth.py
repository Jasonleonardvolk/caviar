# serve_embeddings_noauth.py - Service without authentication
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
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model and cache
model = None
cache = None
embedding_semaphore = None
batch_size = 32

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and cache on startup"""
    global model, cache, embedding_semaphore, batch_size
    
    logger.info("Loading Qwen3-Embedding-8B model...")
    start_time = time.time()
    
    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device=device)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s on device: {model.device}")
    
    # Initialize disk cache
    cache_dir = os.getenv("TORI_EMBED_CACHE", "./emb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = dc.Cache(cache_dir)
    
    # Concurrency control
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(32 * gpu_count)))
    embedding_semaphore = asyncio.Semaphore(batch_size)
    
    logger.info(f"Cache initialized at: {cache_dir}")
    logger.info(f"Embedding semaphore set to: {batch_size}")
    
    yield
    
    # Cleanup
    if model:
        del model
    if cache:
        cache.close()

app = FastAPI(
    title="TORI Embedding Service (No Auth)",
    description="Embedding service using Qwen3-Embedding-8B without authentication",
    version="1.0.0",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Extract and represent the semantic meaning of scientific and technical concepts"
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    cache_hits: int
    cache_misses: int
    processing_time_ms: float

def format_text_with_instruction(text: str, instruction: str) -> str:
    """Format text with instruction prefix for Qwen3"""
    return f"Instruct: {instruction}\nQuery: {text}"

async def embed_cached_concurrent(texts: List[str], instruction: str) -> tuple[List[List[float]], int, int, float]:
    """Concurrent cached embedding"""
    start_time = time.time()
    
    uncached, order, cache_hits, cache_misses = [], [], 0, 0
    
    # Check cache first
    for text in texts:
        cache_key = hashlib.sha256(f"{instruction}::{text}".encode()).hexdigest()
        
        if cache_key in cache:
            order.append(cache[cache_key])
            cache_hits += 1
        else:
            order.append(None)
            uncached.append((cache_key, text))
            cache_misses += 1
    
    # Process uncached texts
    if uncached:
        async with embedding_semaphore:
            keys, texts_to_embed = zip(*uncached)
            formatted_texts = [format_text_with_instruction(text, instruction) for text in texts_to_embed]
            
            with torch.inference_mode():
                embeddings = model.encode(
                    formatted_texts,
                    normalize_embeddings=True,
                    convert_to_tensor=False,
                    show_progress_bar=False
                )
            
            # Cache results
            for key, embedding in zip(keys, embeddings):
                embedding_list = embedding.tolist()
                cache[key] = embedding_list
                none_idx = order.index(None)
                order[none_idx] = embedding_list
    
    processing_time = (time.time() - start_time) * 1000
    return order, cache_hits, cache_misses, processing_time

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embeddings for input texts"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > 1000:
            raise HTTPException(status_code=400, detail="Too many texts (max 1000)")
        
        embeddings, cache_hits, cache_misses, processing_time = await embed_cached_concurrent(
            request.texts, 
            request.instruction
        )
        
        logger.info(f"Processed {len(request.texts)} texts in {processing_time:.2f}ms")
        
        return EmbedResponse(
            embeddings=embeddings,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpu_info = {
                "cuda_available": True,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0)
            }
        except Exception:
            gpu_info = {"error": "GPU metrics unavailable"}
    
    return {
        "status": "healthy",
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": str(model.device) if model else "not_loaded",
        "cache_size": len(cache) if cache else 0,
        "cache_disk_usage_mb": cache.volume() // 1024**2 if cache else 0,
        "system_memory_percent": psutil.virtual_memory().percent,
        "gpu_metrics": gpu_info
    }

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", 8080))
    
    uvicorn.run(
        "serve_embeddings_noauth:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
        access_log=True
    )
