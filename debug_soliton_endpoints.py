#!/usr/bin/env python3
"""
Debug the soliton issues - find out exactly what's wrong
"""

import requests
import json

print("🔍 SOLITON ENDPOINT DEBUGGING")
print("=" * 60)

base_url = "http://localhost:8002/api/soliton"

# Test 1: Check if API is running
print("\n1. Testing API health...")
try:
    r = requests.get("http://localhost:8002/health", timeout=5)
    print(f"   API Health: {r.status_code}")
    if r.status_code == 200:
        print("   ✅ API is running")
except:
    print("   ❌ API is not running!")
    print("   Start it with: python enhanced_launcher.py")
    exit(1)

# Test 2: Check embed endpoint
print("\n2. Testing embed endpoint...")
try:
    r = requests.post(f"{base_url}/embed", 
                     json={"text": "test"}, 
                     timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 404:
        print("   ❌ EMBED ENDPOINT MISSING - This is why you see 'not available'")
        print("   Fix: python restore_and_fix_soliton.py")
    elif r.status_code == 200:
        print("   ✅ Embed endpoint exists")
        data = r.json()
        print(f"   Embedding dimensions: {len(data.get('embedding', []))}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Check stats endpoint with different user formats
print("\n3. Testing stats endpoint variations...")
users_to_test = [
    "adminuser",
    "admin_user", 
    "Admin User",
    "admin%20user",
    "test_user"
]

for user in users_to_test:
    try:
        r = requests.get(f"{base_url}/stats/{user}", timeout=5)
        print(f"\n   User '{user}': Status {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Response: {json.dumps(data, indent=2)}")
            if data.get('status') == 'error':
                print("   ⚠️  Stats returned but with error status")
        else:
            print(f"   Response: {r.text[:100]}")
    except Exception as e:
        print(f"   Error: {e}")

# Test 4: Check diagnostic endpoint
print("\n4. Testing diagnostic endpoint...")
try:
    r = requests.get(f"{base_url}/diagnostic", timeout=5)
    print(f"   Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"   Soliton Available: {data.get('fractal_soliton_available')}")
        print(f"   Active Instances: {data.get('activeInstances', [])}")
except:
    print("   ❌ Diagnostic endpoint not available")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

print("""
The frontend errors show:
1. 'not available' - Usually means embed endpoint returned 404
2. 'Unknown stats error' - Stats endpoint is returning error status

To fix:
1. If embed is missing: python restore_and_fix_soliton.py
2. Restart the API after fixing
3. Check if user 'adminuser' needs to be initialized first
""")
