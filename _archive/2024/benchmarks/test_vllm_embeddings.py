# test_vllm_embeddings.py - Test vLLM embedding endpoint
import requests
import time
import json

def test_vllm():
    url = "http://localhost:8080/v1/embeddings"
    
    # Test payload (OpenAI format)
    payload = {
        "model": "boboliu/Qwen3-Embedding-8B-W4A16-G128",
        "input": [f"Test sentence number {i}." for i in range(10)]
    }
    
    print("🚀 Testing vLLM embeddings...")
    start = time.time()
    
    response = requests.post(url, json=payload)
    elapsed = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        num_embeddings = len(data["data"])
        print(f"✅ Success! Got {num_embeddings} embeddings in {elapsed:.3f}s")
        print(f"⚡ Per text: {elapsed/num_embeddings*1000:.1f}ms")
        print(f"📊 Throughput: {num_embeddings/elapsed:.1f} texts/sec")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_vllm()
