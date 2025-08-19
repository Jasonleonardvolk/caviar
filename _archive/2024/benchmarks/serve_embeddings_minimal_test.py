# serve_embeddings_minimal_test.py - Bare minimum to test raw speed
import torch
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import time
import uvicorn

app = FastAPI()

# Global model
print("Loading model...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
model.eval()
model.max_seq_length = 512  # Short for speed
print(f"Model loaded on {model.device}")

class Request(BaseModel):
    texts: List[str]

@app.post("/embed")
def embed(request: Request):
    # No async, no formatting, just raw embedding
    with torch.inference_mode():
        start = time.time()
        embeddings = model.encode(
            request.texts,
            batch_size=len(request.texts),
            convert_to_numpy=True,
            show_progress_bar=False
        )
        elapsed = time.time() - start
    
    return {
        "embeddings": embeddings.tolist(),
        "time_seconds": elapsed,
        "texts_per_second": len(request.texts) / elapsed
    }

@app.get("/test_batch")
def test_batch():
    # Direct test endpoint
    test_texts = ["Test " + str(i) for i in range(10)]
    
    with torch.inference_mode():
        start = time.time()
        _ = model.encode(test_texts, batch_size=10)
        elapsed = time.time() - start
    
    return {
        "test": "10 texts batched",
        "time_seconds": elapsed,
        "ms_per_text": elapsed * 1000 / 10
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081)  # Different port for testing
