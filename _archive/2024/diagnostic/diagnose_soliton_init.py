#!/usr/bin/env python3
"""
Quick diagnostic for Soliton init error
"""

import requests
import json

print("🔍 Diagnosing Soliton Init Error...\n")

# Test 1: Check if API is responding
try:
    health = requests.get("http://localhost:8002/api/health")
    print(f"✅ API Health: {health.status_code}")
    print(f"   Response: {health.json()}")
except Exception as e:
    print(f"❌ API not responding: {e}")
    exit(1)

# Test 2: Try Soliton init with detailed error capture
print("\n🔍 Testing Soliton init...")
payload = {
    "user_id": "adminuser",
    "lattice_reset": False
}

try:
    response = requests.post(
        "http://localhost:8002/api/soliton/init",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
    
    if response.status_code == 500:
        print("\n⚠️  Getting 500 error - check server logs for:")
        print("  - ImportError in soliton routes")
        print("  - AttributeError in request handling")
        print("  - Missing dependencies")
        
except Exception as e:
    print(f"❌ Request failed: {e}")

# Test 3: Check available endpoints
print("\n🔍 Checking Soliton endpoints...")
try:
    # Try Soliton health
    soliton_health = requests.get("http://localhost:8002/api/soliton/health")
    print(f"Soliton Health: {soliton_health.status_code}")
    if soliton_health.status_code == 200:
        print(f"   Response: {soliton_health.json()}")
except Exception as e:
    print(f"Soliton health check failed: {e}")

print("\n💡 Next steps:")
print("1. Check server console for the actual error")
print("2. The error is likely in prajna_api.py around the soliton_init function")
print("3. Look for ImportError or AttributeError messages")
