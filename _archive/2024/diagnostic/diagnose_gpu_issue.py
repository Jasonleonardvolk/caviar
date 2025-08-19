# diagnose_gpu_issue.py - Find out why embeddings are so slow
import torch
import requests
import numpy as np
import time
import psutil
import GPUtil

print("🔍 DIAGNOSING SLOW EMBEDDING PERFORMANCE")
print("=" * 60)

# 1. Check system resources
print("\n1️⃣ SYSTEM CHECK:")
print(f"CPU Usage: {psutil.cpu_percent()}%")
print(f"RAM Usage: {psutil.virtual_memory().percent}%")
print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")

# 2. Check GPU
print("\n2️⃣ GPU CHECK:")
if torch.cuda.is_available():
    print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")
    
    # Check GPU utilization
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU Load: {gpu.load * 100:.1f}%")
            print(f"GPU Memory: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            print(f"GPU Temp: {gpu.temperature}°C")
    except:
        print("Could not get GPU stats")
else:
    print("❌ CUDA NOT AVAILABLE!")

# 3. Check which service is running
print("\n3️⃣ SERVICE CHECK:")
try:
    response = requests.get("http://localhost:8080/health", timeout=5)
    if response.status_code == 200:
        health = response.json()
        print(f"Service: {health.get('model', 'unknown')}")
        print(f"Device: {health.get('device', 'unknown')}")
        print(f"Max seq length: {health.get('max_seq_length', 'unknown')}")
        
        gpu_info = health.get('gpu_info', {})
        if gpu_info:
            print(f"GPU Memory (from service): {gpu_info.get('memory_allocated_gb', 0):.2f} GB")
except Exception as e:
    print(f"❌ Cannot connect to service: {e}")

# 4. Test raw GPU speed
print("\n4️⃣ RAW GPU SPEED TEST:")
print("Creating test tensors...")

# Test matrix multiplication speed
size = 4096
a = torch.randn(size, size, device='cuda', dtype=torch.float16)
b = torch.randn(size, size, device='cuda', dtype=torch.float16)

# Warmup
torch.cuda.synchronize()
_ = torch.matmul(a, b)
torch.cuda.synchronize()

# Time it
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    _ = torch.matmul(a, b)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Matrix multiply (4096x4096): {elapsed/10*1000:.1f}ms per operation")
print(f"TFLOPS: {(2 * size**3 * 10) / (elapsed * 1e12):.1f}")

# 5. Test instruction formatting overhead
print("\n5️⃣ INSTRUCTION FORMATTING TEST:")
test_text = "This is a test sentence."
instruction = "Extract and represent the semantic meaning"

# Without instruction
start = time.time()
response1 = requests.post("http://localhost:8080/embed", 
                         json={"texts": [test_text]}, 
                         timeout=30)
time1 = time.time() - start

# With instruction
formatted = f"Instruct: {instruction}\nQuery: {test_text}"
start = time.time()
response2 = requests.post("http://localhost:8080/embed", 
                         json={"texts": [formatted]}, 
                         timeout=30)
time2 = time.time() - start

print(f"Without instruction: {time1:.3f}s")
print(f"With instruction: {time2:.3f}s")
print(f"Overhead: {(time2-time1):.3f}s")

print("\n" + "=" * 60)
print("💡 EXPECTED PERFORMANCE:")
print("- Single text: <300ms")
print("- Batch of 10: <1000ms")
print("- Your performance: 3900ms/text 🐌")
print("\n🔧 LIKELY ISSUES:")
print("1. Running on CPU instead of GPU")
print("2. Not using FP16/BF16")
print("3. Instruction formatting adding huge overhead")
print("4. Model loading/unloading each request")
print("5. Wrong service running (not the optimized one)")
