# tests/test_embed_simple.py - Simple embedding test
import requests
import numpy as np

def test_embedding():
    """Simple test without authentication"""
    
    # Test data
    test_texts = ["Koopman operator spectral analysis", "Phase space reconstruction"]
    
    # Make request without auth header
    response = requests.post(
        "http://localhost:8080/embed",
        json={"texts": test_texts}
    )
    
    print(f"Status code: {response.status_code}")
    
    if response.status_code == 401:
        print("❌ Authentication required. Let's check the service configuration...")
        print("Response:", response.json())
        return
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Embedding successful!")
        print(f"  - Embeddings returned: {len(data['embeddings'])}")
        print(f"  - Vector dimensions: {len(data['embeddings'][0])}")
        print(f"  - Cache hits: {data.get('cache_hits', 0)}")
        print(f"  - Cache misses: {data.get('cache_misses', 0)}")
        print(f"  - Processing time: {data.get('processing_time_ms', 0):.2f}ms")
        
        # Check first vector
        vec = np.array(data['embeddings'][0])
        norm = np.linalg.norm(vec)
        print(f"  - First vector norm: {norm:.4f}")
    else:
        print(f"❌ Request failed: {response.status_code}")
        print("Response:", response.text)

if __name__ == "__main__":
    test_embedding()
