#!/usr/bin/env python
"""
Diagnostic script to check if Soliton endpoints are properly registered
"""

import requests
import json
from datetime import datetime

API_BASE = "http://localhost:8002"

print("=" * 60)
print("SOLITON ENDPOINT DIAGNOSTIC")
print(f"Time: {datetime.now()}")
print("=" * 60)

# Test 1: Check API health
print("\n1. Testing API health...")
try:
    response = requests.get(f"{API_BASE}/api/health")
    print(f"✅ API Health: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
except Exception as e:
    print(f"❌ API Health failed: {e}")

# Test 2: List available endpoints
print("\n2. Checking available endpoints...")
try:
    # Try 404 to get endpoint list
    response = requests.get(f"{API_BASE}/nonexistent")
    if response.status_code == 404:
        data = response.json()
        if "available_endpoints" in data:
            endpoints = data["available_endpoints"]
            print(f"✅ Found {len(endpoints)} endpoints:")
            for ep in sorted(endpoints):
                if "soliton" in ep:
                    print(f"   🎯 {ep}")
                else:
                    print(f"   - {ep}")
except Exception as e:
    print(f"❌ Endpoint listing failed: {e}")

# Test 3: Check OpenAPI docs
print("\n3. Checking OpenAPI documentation...")
try:
    response = requests.get(f"{API_BASE}/docs")
    if response.status_code == 200:
        print("✅ API docs available at /docs")
        
        # Get OpenAPI schema
        response = requests.get(f"{API_BASE}/openapi.json")
        if response.status_code == 200:
            openapi = response.json()
            paths = openapi.get("paths", {})
            soliton_paths = [p for p in paths if "soliton" in p]
            print(f"   Found {len(soliton_paths)} soliton endpoints in OpenAPI:")
            for path in sorted(soliton_paths):
                methods = list(paths[path].keys())
                print(f"   - {path} [{', '.join(methods).upper()}]")
except Exception as e:
    print(f"❌ OpenAPI check failed: {e}")

# Test 4: Direct soliton endpoint test
print("\n4. Testing soliton/init endpoint directly...")
try:
    test_data = {"user_id": "test_user"}
    response = requests.post(
        f"{API_BASE}/api/soliton/init",
        json=test_data,
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status: {response.status_code}")
    print(f"   Response: {response.text[:200]}...")
    
    if response.status_code == 200:
        print("✅ Soliton init endpoint is working!")
    else:
        print("⚠️  Soliton init returned non-200 status")
        
except Exception as e:
    print(f"❌ Soliton init test failed: {e}")

# Test 5: Check soliton stats
print("\n5. Testing soliton/stats endpoint...")
try:
    response = requests.get(f"{API_BASE}/api/soliton/stats/adminuser")
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        print(f"   Response: {response.json()}")
        print("✅ Soliton stats endpoint is working!")
    else:
        print(f"   Response: {response.text[:200]}...")
except Exception as e:
    print(f"❌ Soliton stats test failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

print("\n📋 SUMMARY:")
print("If soliton endpoints are not found, check:")
print("1. Is the API server running on port 8002?")
print("2. Are soliton routes properly imported in prajna_api.py?")
print("3. Are routes registered in the startup event?")
print("4. Check the API logs for import errors")
