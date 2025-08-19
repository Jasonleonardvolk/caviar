#!/usr/bin/env python3
"""Test all soliton endpoints"""
import requests
import json
import time

def test_all_endpoints():
    base_url = "http://localhost:8002/api/soliton"
    user_id = f"test_user_{int(time.time())}"
    
    print("\n🧪 TESTING SOLITON ENDPOINTS")
    print("=" * 50)
    
    # Test 1: Health
    print("\n1. Testing /health...")
    try:
        r = requests.get(f"{base_url}/health")
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            print(f"   Response: {json.dumps(r.json(), indent=2)}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Diagnostic
    print("\n2. Testing /diagnostic...")
    try:
        r = requests.get(f"{base_url}/diagnostic")
        print(f"   Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Init
    print(f"\n3. Testing /init with userId={user_id}...")
    try:
        r = requests.post(f"{base_url}/init", json={"userId": user_id})
        print(f"   Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Embed (the missing one!)
    print("\n4. Testing /embed...")
    try:
        r = requests.post(f"{base_url}/embed", json={"text": "Test embedding"})
        print(f"   Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            if "embedding" in data:
                print(f"   ✅ Got embedding with {len(data['embedding'])} dimensions")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Store
    print("\n5. Testing /store...")
    try:
        r = requests.post(f"{base_url}/store", json={
            "user": user_id,
            "concept_id": "test_concept",
            "content": "Test memory",
            "activation_strength": 0.8
        })
        print(f"   Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Stats
    print(f"\n6. Testing /stats/{user_id}...")
    try:
        r = requests.get(f"{base_url}/stats/{user_id}")
        print(f"   Status: {r.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Test complete!")

if __name__ == "__main__":
    test_all_endpoints()
