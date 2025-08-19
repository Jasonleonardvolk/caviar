# serve_embeddings_bnb_correct.py - Correct BitsAndBytes 4-bit implementation
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, models

# Optimize environment
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global embedder
embedder = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder
    
    print("⚡ Loading Qwen3-8B with BitsAndBytes 4-bit (correct method)...")
    start = time.time()
    
    MODEL_ID = "Qwen/Qwen3-Embedding-8B"
    
    # 1️⃣ Bits-and-Bytes config (the new API)
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True  # Extra compression
    )
    
    # 2️⃣ Load tokenizer + HF model with that config
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    
    print("Loading 4-bit quantized model...")
    hfmdl = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True
    )
    
    # 3️⃣ Wrap in Sentence-Transformers modules
    print("Creating SentenceTransformer wrapper...")
    st_transformer = models.Transformer(
        model_name_or_path=MODEL_ID,
        tokenizer=tok,
        model=hfmdl,  # Pass the pre-loaded 4-bit model
        max_seq_length=512
    )
    
    pooling = models.Pooling(
        word_embedding_dimension=hfmdl.config.hidden_size,
        pooling_mode_cls_token=True,  # Qwen uses CLS token
        pooling_mode_mean_tokens=False,
        pooling_mode_max_tokens=False
    )
    
    embedder = SentenceTransformer(modules=[st_transformer, pooling], device="cuda")
    embedder.eval()
    
    # 4️⃣ Warmup test
    print("Warming up...")
    with torch.inference_mode():
        t0 = time.time()
        vecs = embedder.encode(
            ["warmup"] * 10,
            batch_size=10,
            show_progress_bar=False
        )
        warmup_time = time.time() - t0
    
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    print(f"⚡ Warmup: 10 texts in {warmup_time:.3f}s (dim = {vecs.shape[1]})")
    print(f"💾 Using 4-bit quantization - ~10GB VRAM")
    print(f"🚀 Ready! Expecting ~700ms for 10 texts")
    
    yield
    
    del embedder
    torch.cuda.empty_cache()

app = FastAPI(
    title="TORI BnB 4-bit Service (Correct)",
    lifespan=lifespan
)

class EmbedRequest(BaseModel):
    texts: List[str]
    instruction: str = "Extract and represent the semantic meaning"

@app.post("/embed")
async def embed(request: EmbedRequest):
    start = time.time()
    
    # Format with instruction
    formatted_texts = [f"Instruct: {request.instruction}\nQuery: {text}" 
                      for text in request.texts]
    
    # Embed with proper batching
    with torch.inference_mode():
        embeddings = embedder.encode(
            formatted_texts,
            batch_size=min(32, len(formatted_texts)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
    
    elapsed = time.time() - start
    ms = elapsed * 1000
    
    print(f"⚡ Processed {len(request.texts)} texts in {ms:.1f}ms ({ms/len(request.texts):.1f}ms per text)")
    
    return {
        "embeddings": embeddings.tolist(),
        "processing_time_ms": ms,
        "texts_processed": len(request.texts),
        "cache_hits": 0,
        "cache_misses": len(request.texts)
    }

@app.get("/health")
async def health():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "cuda_available": True,
            "device_name": torch.cuda.get_device_name(0),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
        }
    
    return {
        "status": "healthy",
        "model": "Qwen3-8B",
        "quantization": "BitsAndBytes 4-bit NF4",
        "max_seq_length": 512,
        "gpu_info": gpu_info
    }

if __name__ == "__main__":
    # av.logging fix
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
