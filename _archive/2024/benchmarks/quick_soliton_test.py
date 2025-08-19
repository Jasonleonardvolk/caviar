#!/usr/bin/env python
"""
Quick test to verify Soliton endpoints are working
"""
import requests
import json

print("Testing Soliton Endpoints...")
print("-" * 40)

# Test 1: Init endpoint
print("\n1. Testing /api/soliton/init")
try:
    response = requests.post(
        "http://localhost:8002/api/soliton/init",
        json={"user_id": "test_user"},
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        print("   ✅ Init endpoint is working!")
    else:
        print(f"   ❌ Error: {response.text}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

# Test 2: Stats endpoint
print("\n2. Testing /api/soliton/stats/adminuser")
try:
    response = requests.get("http://localhost:8002/api/soliton/stats/adminuser")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        print("   ✅ Stats endpoint is working!")
    else:
        print(f"   ❌ Error: {response.text}")
except Exception as e:
    print(f"   ❌ Failed: {e}")

print("\n" + "-" * 40)
print("Test complete!")

# Additional check - list all routes
print("\n3. Checking available routes...")
try:
    response = requests.get("http://localhost:8002/openapi.json")
    if response.status_code == 200:
        openapi = response.json()
        paths = openapi.get("paths", {})
        soliton_paths = [p for p in paths if "soliton" in p]
        if soliton_paths:
            print(f"   Found {len(soliton_paths)} soliton endpoints:")
            for path in sorted(soliton_paths):
                print(f"   - {path}")
        else:
            print("   ⚠️  No soliton endpoints found in OpenAPI!")
except Exception as e:
    print(f"   Failed to check OpenAPI: {e}")
