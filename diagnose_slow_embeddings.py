# diagnose_slow_embeddings.py - Find out why embeddings are slow
import asyncio
import time
import httpx
import psutil
import torch

async def diagnose_performance():
    """Diagnose why embeddings are slow"""
    
    print("🔍 Diagnosing embedding performance issues...\n")
    
    # Check system resources
    print("💻 System Resources:")
    print(f"  - CPU Usage: {psutil.cpu_percent()}%")
    print(f"  - Memory Usage: {psutil.virtual_memory().percent}%")
    print(f"  - Available Memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    
    if torch.cuda.is_available():
        print(f"\n🎮 GPU Status:")
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - GPU Memory Used: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
        print(f"  - GPU Memory Cached: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")
    
    # Test different text lengths
    print("\n⏱️ Testing embedding performance...")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        # Test 1: Single short text
        print("\n1. Single short text:")
        start = time.time()
        response = await client.post(
            "http://localhost:8080/embed",
            json={"texts": ["This is a short test."]}
        )
        elapsed = time.time() - start
        print(f"  - Status: {response.status_code}")
        print(f"  - Time: {elapsed:.2f}s")
        if response.status_code == 200:
            data = response.json()
            print(f"  - Processing time (server): {data.get('processing_time_ms', 0)/1000:.2f}s")
        
        # Test 2: Multiple short texts
        print("\n2. Five short texts:")
        texts = [f"Test sentence number {i}." for i in range(5)]
        start = time.time()
        response = await client.post(
            "http://localhost:8080/embed",
            json={"texts": texts}
        )
        elapsed = time.time() - start
        print(f"  - Status: {response.status_code}")
        print(f"  - Time: {elapsed:.2f}s")
        print(f"  - Time per text: {elapsed/5:.2f}s")
        
        # Test 3: Single long text
        print("\n3. Single long text (1000 chars):")
        long_text = "This is a test. " * 62  # ~1000 chars
        start = time.time()
        response = await client.post(
            "http://localhost:8080/embed",
            json={"texts": [long_text]}
        )
        elapsed = time.time() - start
        print(f"  - Status: {response.status_code}")
        print(f"  - Time: {elapsed:.2f}s")
        
        # Test 4: Check cache
        print("\n4. Cached text (repeat of test 1):")
        start = time.time()
        response = await client.post(
            "http://localhost:8080/embed",
            json={"texts": ["This is a short test."]}
        )
        elapsed = time.time() - start
        print(f"  - Status: {response.status_code}")
        print(f"  - Time: {elapsed:.2f}s")
        if response.status_code == 200:
            data = response.json()
            print(f"  - Cache hits: {data.get('cache_hits', 0)}")
            print(f"  - Cache misses: {data.get('cache_misses', 0)}")

async def test_direct_embedding():
    """Test embedding without going through the service"""
    print("\n🔍 Testing direct model performance...")
    
    from sentence_transformers import SentenceTransformer
    import torch
    
    # Load model
    print("Loading model...")
    start = time.time()
    model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model loaded in {time.time() - start:.2f}s")
    
    # Test embedding
    test_texts = ["Test sentence 1", "Test sentence 2", "Test sentence 3"]
    
    print("\nGenerating embeddings...")
    start = time.time()
    with torch.no_grad():
        embeddings = model.encode(test_texts, show_progress_bar=True)
    elapsed = time.time() - start
    
    print(f"✅ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    print(f"   Time per embedding: {elapsed/len(test_texts):.2f}s")

if __name__ == "__main__":
    print("=== Embedding Performance Diagnosis ===\n")
    asyncio.run(diagnose_performance())
    
    # Also test direct
    print("\n" + "="*40)
    asyncio.run(test_direct_embedding())
