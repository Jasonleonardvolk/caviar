#!/usr/bin/env python3
"""
Test all soliton endpoints to verify everything is working
"""
import requests
import json
import time

def test_soliton_endpoints():
    base_url = "http://localhost:8002/api/soliton"
    
    print("\n" + "="*60)
    print("TESTING SOLITON ENDPOINTS")
    print("="*60 + "\n")
    
    endpoints = [
        ("GET", "/health", None),
        ("GET", "/diagnostic", None),
        ("POST", "/init", {"userId": "test_user"}),
        ("POST", "/embed", {"text": "Test embedding generation"}),
        ("GET", "/stats/testuser", None),
    ]
    
    passed = 0
    failed = 0
    
    for method, endpoint, data in endpoints:
        url = base_url + endpoint
        print(f"\nTesting {method} {endpoint}...")
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=5)
            else:
                response = requests.post(url, json=data, timeout=5)
            
            if response.status_code == 200:
                print(f"  ✅ SUCCESS - Status: {response.status_code}")
                result = response.json()
                if endpoint == "/embed" and "embedding" in result:
                    print(f"  ✅ Got embedding with {len(result['embedding'])} dimensions")
                passed += 1
            else:
                print(f"  ❌ FAILED - Status: {response.status_code}")
                try:
                    print(f"  Response: {response.json()}")
                except:
                    print(f"  Response: {response.text[:200]}")
                failed += 1
                
        except requests.exceptions.ConnectionError:
            print("  ❌ CONNECTION FAILED - Is the API running?")
            print("  Start with: python main.py")
            failed += 1
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✅ All soliton endpoints are working!")
        print("The embed endpoint fix was successful!")
    else:
        print("\n⚠️  Some endpoints failed. Check the errors above.")
        
        # Check if it's just a connection issue
        if all("CONNECTION FAILED" in str(e) for e in [failed]):
            print("\n💡 TIP: Make sure the API is running:")
            print("   python main.py")

if __name__ == "__main__":
    print("🔍 Soliton Endpoint Verification")
    print("This will test all soliton endpoints including the fixed embed endpoint")
    
    test_soliton_endpoints()
