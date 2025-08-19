# test_batch_performance.py - Test batching performance
import requests
import time
import json

def test_batching():
    url = "http://127.0.0.1:8080/embed"
    
    # Test different batch sizes
    for batch_size in [1, 5, 10, 32]:
        texts = [f"Test sentence number {i} for batch testing." for i in range(batch_size)]
        payload = {"texts": texts}
        
        print(f"\nTesting batch size {batch_size}:")
        t0 = time.perf_counter()
        
        try:
            r = requests.post(url, json=payload, timeout=60)
            elapsed = time.perf_counter() - t0
            
            if r.status_code == 200:
                data = r.json()
                print(f"  ✅ Success in {elapsed:.3f}s")
                print(f"  - Server processing: {data['processing_time_ms']:.1f}ms")
                print(f"  - Per text: {data['processing_time_ms']/batch_size:.1f}ms")
                print(f"  - Cache hits: {data['cache_hits']}, misses: {data['cache_misses']}")
            else:
                print(f"  ❌ Failed: {r.status_code}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Testing Embedding Batch Performance")
    print("=" * 40)
    test_batching()
