# benchmark_embedding_speed.py - Compare embedding speeds
import time
import requests
import json

def benchmark_speed():
    url = "http://127.0.0.1:8080/embed"
    
    print("⚡ EMBEDDING SPEED BENCHMARK")
    print("=" * 50)
    
    # Test different batch sizes
    test_cases = [
        (1, "Single sentence"),
        (5, "Five sentences"),
        (10, "Ten sentences"),
        (32, "Thirty-two sentences")
    ]
    
    for batch_size, description in test_cases:
        texts = [f"This is test sentence number {i} for benchmarking embedding speed." 
                 for i in range(batch_size)]
        
        payload = {"texts": texts}
        
        print(f"\n📊 {description} (batch size: {batch_size}):")
        
        # Time the request
        start = time.time()
        try:
            response = requests.post(url, json=payload, timeout=60)
            total_time = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                server_time = data.get('processing_time_ms', 0)
                
                print(f"  ✅ Total time: {total_time:.3f}s")
                print(f"  ⚡ Server processing: {server_time:.1f}ms")
                print(f"  📈 Per text: {server_time/batch_size:.1f}ms")
                
                # Calculate throughput
                throughput = batch_size / (server_time / 1000)
                print(f"  🚀 Throughput: {throughput:.1f} texts/second")
            else:
                print(f"  ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("💡 With batching + inference mode, expect:")
    print("   - Single text: ~150-300ms")
    print("   - Batch of 32: ~600-1000ms total (~20-30ms per text)")
    print("   - Throughput: 30-50 texts/second")

if __name__ == "__main__":
    benchmark_speed()
