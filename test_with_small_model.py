# test_with_small_model.py - Test with a small model to verify setup
import torch
import time
from sentence_transformers import SentenceTransformer

print("🔍 TESTING WITH SMALLER MODEL")
print("=" * 60)

# Try a much smaller model
print("\n1️⃣ Loading all-MiniLM-L6-v2 (22M params vs 8B)...")
start = time.time()
small_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
small_model.eval()
print(f"Model loaded in {time.time() - start:.2f}s")

# Test texts
test_texts = [f"Test sentence {i}." for i in range(10)]

print("\n2️⃣ Testing small model speed:")
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.time()
    embeddings = small_model.encode(test_texts, batch_size=10, show_progress_bar=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s ({elapsed/10*1000:.1f}ms per text)")
print(f"Embeddings shape: {embeddings.shape}")

# Test with 100 texts
print("\n3️⃣ Testing with 100 texts:")
many_texts = [f"Text {i}" for i in range(100)]
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.time()
    embeddings = small_model.encode(many_texts, batch_size=32, show_progress_bar=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start

print(f"Time: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per text)")
print(f"Speed: {100/elapsed:.1f} texts/second")

print("\n" + "=" * 60)
print("If this small model is fast (<10ms per text), your GPU setup is fine.")
print("The issue is specifically with the large Qwen3-8B model.")
