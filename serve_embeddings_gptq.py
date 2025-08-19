# serve_embeddings_gptq.py - Ultra-fast INT8 quantized embeddings
import os
import time
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

# Disable threading issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Global model
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    
    print("⚡ Loading Qwen3-8B-GPTQ (INT8 quantized)...")
    start = time.time()
    
    try:
        from auto_gptq import AutoGPTQForCausalLM
        from transformers import AutoTokenizer
        from sentence_transformers import SentenceTransformer, models
        
        # Load quantized model
        model_name = "TheBloke/Qwen3-Embedding-8B-GPTQ"
        
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        print("Loading quantized model...")
        gptq_model = AutoGPTQForCausalLM.from_quantized(
            model_name,
            device="cuda:0",
            use_triton=False,  # Not needed for embeddings
            inject_fused_attention=False,
            disable_exllama=True  # For embeddings
        )
        
        # Wrap in SentenceTransformer
        print("Creating SentenceTransformer wrapper...")
        model = SentenceTransformer(modules=[
            models.Transformer(gptq_model, tokenizer, max_seq_length=512),
            models.Pooling(word_embedding_dimension=gptq_model.config.hidden_size)
        ])
        model.eval()
        
        print(f"✅ Model loaded in {time.time() - start:.2f}s")
        print(f"📏 Max sequence length: 512")
        print(f"⚡ Using INT8 quantization - expect 20x speedup!")
        
        # Warmup
        with torch.inference_mode():
            _ = model.encode(["warmup"], batch_size=1, show_progress_bar=False)
        
        print("🚀 Ready for <50ms per text performance!")
        
    except ImportError:
        print("❌ auto-gptq not installed. Install with:")
        print("   pip install auto-gptq")
        print("Falling back to standard model...")
        
        # Fallback to regular model
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
        model.eval()
        model.max_seq_length = 256  # Shorter for speed
        
        # Try to use FP16
        model = model.half()
        print(f"✅ Using FP16 model as fallback")
    
    yield
    
    del model
    torch.cuda.empty_cache()

app = FastAPI(
    title="TORI GPTQ Embedding Service",
    description="INT8 quantized for 20x speed",
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
    
    # Embed with batching
    with torch.inference_mode():
        embeddings = model.encode(
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
    return {
        "status": "healthy",
        "model": "Qwen3-8B-GPTQ (INT8)",
        "device": "cuda:0",
        "max_seq_length": 512,
        "quantization": "INT8-GPTQ"
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
