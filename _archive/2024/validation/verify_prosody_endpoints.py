"""
Verify Prosody Engine Endpoints
"""
import requests
import json

BASE_URL = "http://localhost:8080"  # Prajna runs on 8080

print("🔍 VERIFYING PROSODY ENGINE ENDPOINTS...")
print("-" * 50)

# Test endpoints
endpoints = [
    ("/api/prosody/health", "GET"),
    ("/api/prosody/emotions/list?limit=5", "GET"),
]

for endpoint, method in endpoints:
    try:
        url = BASE_URL + endpoint
        print(f"\nTesting: {method} {endpoint}")
        
        if method == "GET":
            response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            print(f"✅ SUCCESS: {endpoint}")
            print(f"Response: {json.dumps(response.json(), indent=2)[:200]}...")
        else:
            print(f"❌ FAILED: {endpoint} - Status: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Cannot connect to {BASE_URL} - Is Prajna running?")
        break
    except Exception as e:
        print(f"❌ ERROR: {endpoint} - {e}")

print("\n" + "-" * 50)
print("💡 TIP: Start TORI with: poetry run python enhanced_launcher.py")
