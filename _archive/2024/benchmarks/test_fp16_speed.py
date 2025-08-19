# test_fp16_speed.py - Test if FP16 makes it faster
import torch
import time
from sentence_transformers import SentenceTransformer

print("🔍 TESTING FP16 vs FP32 SPEED")
print("=" * 60)

# Load model
print("\n1️⃣ Loading model in default mode...")
model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
model.eval()
model.max_seq_length = 512

# Check current dtype
print(f"Model dtype: {next(model.parameters()).dtype}")

# Test texts
test_texts = [f"Test sentence {i}." for i in range(10)]

# Test 1: Default precision
print("\n2️⃣ Testing with default precision:")
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.time()
    embeddings1 = model.encode(test_texts, batch_size=10, show_progress_bar=False)
    torch.cuda.synchronize()
    time1 = time.time() - start
print(f"Time: {time1:.3f}s ({time1/10*1000:.1f}ms per text)")

# Convert to FP16
print("\n3️⃣ Converting model to FP16...")
model = model.half()
print(f"Model dtype after conversion: {next(model.parameters()).dtype}")

# Test 2: FP16 precision
print("\n4️⃣ Testing with FP16 precision:")
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.time()
    embeddings2 = model.encode(test_texts, batch_size=10, show_progress_bar=False)
    torch.cuda.synchronize()
    time2 = time.time() - start
print(f"Time: {time2:.3f}s ({time2/10*1000:.1f}ms per text)")

print("\n5️⃣ Testing single long text (2K chars):")
long_text = "This is a long test. " * 100  # ~2000 chars
with torch.inference_mode():
    torch.cuda.synchronize()
    start = time.time()
    _ = model.encode([long_text], batch_size=1, show_progress_bar=False)
    torch.cuda.synchronize()
    time3 = time.time() - start
print(f"Time: {time3:.3f}s")

print("\n" + "=" * 60)
print("ANALYSIS:")
print(f"FP32 time: {time1:.3f}s")
print(f"FP16 time: {time2:.3f}s")
print(f"Speedup: {time1/time2:.1f}x")

if time2 < time1 * 0.7:
    print("\n✅ FP16 provides significant speedup!")
else:
    print("\n⚠️ FP16 doesn't help much - issue is elsewhere")
