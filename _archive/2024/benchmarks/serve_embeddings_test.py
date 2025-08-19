# serve_embeddings_test.py - Test version with smaller model
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

# Global model and cache
model = None
cache = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model and cache on startup"""
    global model, cache
    
    logger.info("Loading embedding model...")
    start_time = time.time()
    
    # Use a smaller, well-supported model for testing
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "all-MiniLM-L6-v2"  # Small, fast, reliable model
    
    logger.info(f"Loading {model_name} on {device}...")
    model = SentenceTransformer(model_name, device=device)
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s on device: {model.device}")
    
    # Initialize disk cache
    cache_dir = "./emb_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache = dc.Cache(cache_dir)
    
    logger.info(f"Cache initialized at: {cache_dir}")
    
    yield
    
    # Cleanup
    if model:
        del model
    if cache:
        cache.close()

app = FastAPI(
    title="TORI Test Embedding Service",
    description="Test embedding service with MiniLM model",
    version="1.0.0",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    cache_hits: int
    cache_misses: int

def embed_cached(texts: List[str]) -> tuple[List[List[float]], int, int]:
    """Cached embedding"""
    uncached, order, cache_hits, cache_misses = [], [], 0, 0
    
    for text in texts:
        cache_key = hashlib.sha256(text.encode()).hexdigest()
        
        if cache_key in cache:
            order.append(cache[cache_key])
            cache_hits += 1
        else:
            order.append(None)
            uncached.append((cache_key, text))
            cache_misses += 1
    
    if uncached:
        keys, texts_to_embed = zip(*uncached)
        
        with torch.inference_mode():
            embeddings = model.encode(
                texts_to_embed,
                normalize_embeddings=True,
                convert_to_tensor=False
            )
        
        for key, embedding in zip(keys, embeddings):
            cache[key] = embedding.tolist()
            none_idx = order.index(None)
            order[none_idx] = embedding.tolist()
    
    return order, cache_hits, cache_misses

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embeddings for input texts"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        embeddings, cache_hits, cache_misses = embed_cached(request.texts)
        
        return EmbedResponse(
            embeddings=embeddings,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )
        
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "all-MiniLM-L6-v2",
        "device": str(model.device) if model else "not_loaded",
        "cache_size": len(cache) if cache else 0
    }

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", 8080))
    
    uvicorn.run(
        "serve_embeddings_test:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        reload=False,
        access_log=True
    )
