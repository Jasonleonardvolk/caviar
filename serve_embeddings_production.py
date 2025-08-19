# serve_embeddings_production.py - Production TORI embedding server
import os
import asyncio
import logging
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
    
    logger.info("Loading Qwen3-Embedding-8B model...")
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "torch_dtype": torch.float16
        },
        tokenizer_kwargs={"padding_side": "left"}
    )
    
    # Initialize disk cache
    cache_dir = os.getenv("TORI_EMBED_CACHE", "/var/tori/emb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = dc.Cache(cache_dir)
    
    logger.info(f"Model loaded on device: {model.device}")
    logger.info(f"Cache initialized at: {cache_dir}")
    
    yield
    
    # Cleanup
    if model:
        del model
    if cache:
        cache.close()

app = FastAPI(
    title="TORI Embedding Service",
    description="High-performance embedding service using Qwen3-Embedding-8B",
    version="1.0.0",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Given a document, extract key concepts and semantic relationships"
    normalize: bool = True

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    cache_hits: int
    cache_misses: int

def format_text_with_instruction(text: str, instruction: str) -> str:
    """Format text with instruction prefix for Qwen3"""
    return f"Instruct: {instruction}\nQuery: {text}"

def embed_cached(texts: List[str], instruction: str) -> tuple[List[List[float]], int, int]:
    """Cached embedding with SHA-based lookup"""
    uncached, order, cache_hits, cache_misses = [], [], 0, 0
    
    for text in texts:
        # Create cache key from instruction + text
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
        keys, texts_to_embed = zip(*uncached)
        formatted_texts = [format_text_with_instruction(text, instruction) for text in texts_to_embed]
        
        with torch.inference_mode():
            embeddings = model.encode(
                formatted_texts,
                normalize_embeddings=True,
                convert_to_tensor=False
            )
        
        # Cache results and fill order
        for key, embedding in zip(keys, embeddings):
            cache[key] = embedding.tolist()
            # Find and replace the None placeholder
            none_idx = order.index(None)
            order[none_idx] = embedding.tolist()
    
    return order, cache_hits, cache_misses

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(request: EmbedRequest):
    """Generate embeddings for input texts"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > 1000:
            raise HTTPException(status_code=400, detail="Too many texts (max 1000)")
        
        embeddings, cache_hits, cache_misses = embed_cached(
            request.texts, 
            request.instruction
        )
        
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
        "model": "Qwen/Qwen3-Embedding-8B",
        "device": str(model.device) if model else "not_loaded",
        "cache_size": len(cache) if cache else 0
    }

@app.get("/stats")
async def embedding_stats():
    """Get model and cache statistics"""
    return {
        "model_info": {
            "name": "Qwen/Qwen3-Embedding-8B",
            "device": str(model.device) if model else "not_loaded",
            "max_seq_length": getattr(model, 'max_seq_length', 'unknown') if model else 'unknown'
        },
        "cache_info": {
            "size": len(cache) if cache else 0,
            "directory": cache.directory if cache else None
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", 8080))
    workers = int(os.getenv("EMBED_WORKERS", 1))
    
    uvicorn.run(
        "serve_embeddings:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=False,
        access_log=True
    )
