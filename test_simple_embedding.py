# test_simple_embedding.py - Test basic embedding functionality
import requests
import json

def test_embedding():
    # Test with simple text
    url = "http://localhost:8080/embed"
    
    print("Testing embedding service...")
    
    # Test 1: Single simple text
    payload = {"texts": ["This is a simple test."]}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"  - Embeddings: {len(data.get('embeddings', []))}")
            print(f"  - Dimensions: {len(data['embeddings'][0]) if data.get('embeddings') else 0}")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Multiple texts
    print("\nTesting with multiple texts...")
    payload = {"texts": ["First sentence.", "Second sentence.", "Third sentence."]}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! Got {len(data.get('embeddings', []))} embeddings")
        else:
            print(f"❌ Failed: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_embedding()
