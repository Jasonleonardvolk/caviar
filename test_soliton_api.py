"""
Test script for Soliton API endpoints
Run this after starting the backend to verify the fix works
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000/api/soliton"

def test_soliton_api():
    print("Testing Soliton API Endpoints...")
    print("=" * 40)
    
    # 1. Initialize
    print("\n1. Testing /initialize endpoint...")
    response = requests.post(f"{BASE_URL}/initialize", json={
        "user_id": "test_user_123",
        "lattice_reset": False
    })
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Give it a moment to initialize
    time.sleep(1)
    
    # 2. Store a memory with embedding
    print("\n2. Testing /store endpoint...")
    response = requests.post(f"{BASE_URL}/store", json={
        "user_id": "test_user_123",
        "concept_id": "test_concept_001",
        "content": {"text": "This is a test memory", "type": "test"},
        "activation_strength": 0.8,
        "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    })
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Store another memory for better testing
    response = requests.post(f"{BASE_URL}/store", json={
        "user_id": "test_user_123",
        "concept_id": "test_concept_002",
        "content": {"text": "Another test memory", "type": "test"},
        "activation_strength": 0.9,
        "embedding": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    })
    
    # 3. Query memories - This is the critical test!
    print("\n3. Testing /query endpoint (THIS SHOULD NOT GIVE 500 ERROR)...")
    response = requests.post(f"{BASE_URL}/query", json={
        "user_id": "test_user_123",
        "query_embedding": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
        "k": 5
    })
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ SUCCESS! Query endpoint is working!")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print("❌ FAILED! Still getting errors.")
        print(f"Error: {response.text}")
    
    # 4. Get stats
    print("\n4. Testing /stats endpoint...")
    response = requests.get(f"{BASE_URL}/stats/test_user_123")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # 5. Check health
    print("\n5. Testing /health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    print("\n" + "=" * 40)
    print("Test completed!")

if __name__ == "__main__":
    try:
        test_soliton_api()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to the API.")
        print("Make sure the backend is running on http://localhost:8000")
        print("Run: python enhanced_launcher.py")
    except Exception as e:
        print(f"❌ ERROR: {e}")
