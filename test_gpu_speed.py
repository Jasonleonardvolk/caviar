# test_gpu_speed.py - Test raw GPU embedding speed
import torch
import time
from sentence_transformers import SentenceTransformer

print("Testing raw GPU embedding speed...")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load model
print("\nLoading model...")
start = time.time()
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
model.eval()
model.max_seq_length = 512
print(f"Model loaded in {time.time() - start:.2f}s")

# Test different batch sizes
test_texts = [f"This is test sentence number {i}." for i in range(32)]

print("\nTesting embedding speed:")
for batch_size in [1, 4, 8, 16, 32]:
    texts = test_texts[:batch_size]
    
    # Warmup
    with torch.inference_mode():
        _ = model.encode(texts, show_progress_bar=False)
    
    # Time it
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.inference_mode():
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Batch {batch_size:2d}: {elapsed*1000:6.1f}ms total, {elapsed*1000/batch_size:6.1f}ms per text")

# Check GPU memory
print(f"\nGPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated, {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
