# serve_embeddings_production_final.py - Production-ready with all fixes
import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import jwt
from prometheus_fastapi_instrumentator import Instrumentator
from sentence_transformers import SentenceTransformer
import hashlib
import diskcache as dc
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key")
MAX_REQUESTS_PER_MINUTE = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "64"))

# Global model, cache, and concurrency control
model = None
cache = None
embedding_semaphore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model, cache, and semaphore on startup"""
    global model, cache, embedding_semaphore
    
    logger.info("Loading Qwen3-Embedding-8B model...")
    start_time = time.time()
    
    # Load model with optimizations
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "device_map": "auto",
            "torch_dtype": torch.float16
        },
        tokenizer_kwargs={"padding_side": "left"}
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s on device: {model.device}")
    
    # Initialize disk cache with eviction policy
    cache_dir = os.getenv("TORI_EMBED_CACHE", "/var/tori/emb_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache = dc.Cache(
        cache_dir,
        size_limit=int(os.getenv("CACHE_SIZE_GB", "10")) * 1024**3,  # 10GB default
        disk_min_file_size=1024  # Avoid tiny file spam
    )
    
    # Concurrency control - batch size based on GPU memory
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
    title="TORI Production Embedding Service",
    description="High-performance embedding service using Qwen3-Embedding-8B",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Restrict to your frontend
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Add Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Rate limiting (simple in-memory, use Redis for production scale)
request_counts = {}
rate_limit_window = 60  # 1 minute

def verify_jwt_token(token: str = Depends(security)) -> dict:
    """Verify JWT token for authentication"""
    try:
        if os.getenv("DISABLE_AUTH", "false").lower() == "true":
            return {"user": "dev"}
        
        payload = jwt.decode(token.credentials, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

async def rate_limit_check(user_data: dict = Depends(verify_jwt_token)):
    """Simple rate limiting"""
    user_id = user_data.get("user", "anonymous")
    current_time = time.time()
    
    # Clean old entries
    cutoff = current_time - rate_limit_window
    request_counts[user_id] = [t for t in request_counts.get(user_id, []) if t > cutoff]
    
    # Check rate limit
    if len(request_counts[user_id]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )
    
    # Record request
    request_counts[user_id].append(current_time)
    return user_data

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
    """Concurrent cached embedding with semaphore control"""
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
    
    # Process uncached texts with concurrency control
    if uncached:
        async with embedding_semaphore:
            keys, texts_to_embed = zip(*uncached)
            formatted_texts = [format_text_with_instruction(text, instruction) for text in texts_to_embed]
            
            with torch.inference_mode():
                # Process in batches to avoid GPU OOM
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(formatted_texts), batch_size):
                    batch = formatted_texts[i:i + batch_size]
                    batch_embeddings = model.encode(
                        batch,
                        normalize_embeddings=True,  # Only normalize here, not in fallback
                        convert_to_tensor=False,
                        show_progress_bar=False
                    )
                    all_embeddings.extend(batch_embeddings)
            
            # Cache results and fill order
            for key, embedding in zip(keys, all_embeddings):
                embedding_list = embedding.tolist()
                cache[key] = embedding_list
                # Find and replace the None placeholder
                none_idx = order.index(None)
                order[none_idx] = embedding_list
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    return order, cache_hits, cache_misses, processing_time

@app.post("/embed", response_model=EmbedResponse)
async def embed_text(
    request: EmbedRequest,
    user_data: dict = Depends(rate_limit_check)
):
    """Generate embeddings for input texts with rate limiting and auth"""
    try:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        
        if len(request.texts) > 1000:
            raise HTTPException(status_code=400, detail="Too many texts (max 1000)")
        
        embeddings, cache_hits, cache_misses, processing_time = await embed_cached_concurrent(
            request.texts, 
            request.instruction
        )
        
        logger.info(f"Processed {len(request.texts)} texts for user {user_data.get('user', 'unknown')} in {processing_time:.2f}ms")
        
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
    """Enhanced health check with GPU metrics"""
    gpu_info = {}
    if torch.cuda.is_available():
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                gpu_info[f"gpu_{i}"] = {
                    "utilization": f"{gpu.load * 100:.1f}%",
                    "memory_used": f"{gpu.memoryUsed}MB",
                    "memory_total": f"{gpu.memoryTotal}MB",
                    "temperature": f"{gpu.temperature}°C"
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

@app.get("/stats")
async def embedding_stats():
    """Comprehensive statistics"""
    return {
        "model_info": {
            "name": "Qwen/Qwen3-Embedding-8B",
            "device": str(model.device) if model else "not_loaded",
            "max_seq_length": getattr(model, 'max_seq_length', 'unknown') if model else 'unknown'
        },
        "cache_info": {
            "size": len(cache) if cache else 0,
            "directory": cache.directory if cache else None,
            "disk_usage_bytes": cache.volume() if cache else 0
        },
        "concurrency": {
            "semaphore_capacity": embedding_semaphore._value if embedding_semaphore else 0,
            "current_requests": f"{embedding_semaphore._value}/{batch_size}" if embedding_semaphore else "unknown"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("EMBED_PORT", 8080))
    workers = 1  # Single worker due to global model state
    
    uvicorn.run(
        "serve_embeddings_production_final:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        reload=False,
        access_log=True
    )
