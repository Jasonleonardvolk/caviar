"""
Simple test to check if Soliton API is working or returning 500 errors
"""

import requests
import json

print("\n" + "="*50)
print("TESTING SOLITON API")
print("="*50 + "\n")

base_url = "http://localhost:5173"

# Test 1: Health Check
print("1. Testing /api/soliton/health...")
try:
    response = requests.get(f"{base_url}/api/soliton/health")
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"   [OK] Response: {response.json()}")
    elif response.status_code == 500:
        print("   [ERROR] Got 500 - THIS IS THE ISSUE WE'RE FIXING!")
        print(f"   Error: {response.text[:200]}...")
except Exception as e:
    print(f"   [ERROR] Connection failed: {e}")
    print("   Is the API running? Start it with:")
    print("   uvicorn api.main:app --port 5173 --reload")

# Test 2: Diagnostic (might not exist yet)
print("\n2. Testing /api/soliton/diagnostic...")
try:
    response = requests.get(f"{base_url}/api/soliton/diagnostic")
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 200:
        diag = response.json()
        print(f"   [OK] Soliton Available: {diag.get('soliton_available', 'unknown')}")
    elif response.status_code == 404:
        print("   [INFO] Diagnostic endpoint not found (fix not applied yet)")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 3: Init
print("\n3. Testing /api/soliton/init...")
try:
    response = requests.post(
        f"{base_url}/api/soliton/init",
        json={"userId": "test_user"},
        headers={"Content-Type": "application/json"}
    )
    print(f"   Status Code: {response.status_code}")
    if response.status_code == 200:
        print(f"   [OK] Response: {response.json()}")
    elif response.status_code == 500:
        print("   [ERROR] Got 500 - THIS IS THE ISSUE WE'RE FIXING!")
        print(f"   Error: {response.text[:200]}...")
except Exception as e:
    print(f"   [ERROR] {e}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print("\nIf you see 500 errors above:")
print("1. Run the fix: .\\RUN_GREMLIN_FIX.ps1")
print("2. Or manually: .\\GREMLIN_HUNTER_MASTER_FIXED.ps1")
print("\nIf connection failed, start the API first:")
print("cd C:\\Users\\jason\\Desktop\\tori\\kha")
print("uvicorn api.main:app --port 5173 --reload\n")

input("Press Enter to exit...")
