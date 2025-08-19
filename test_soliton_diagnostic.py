#!/usr/bin/env python3
"""
Diagnostic script to test soliton initialization and identify the exact error
"""

import requests
import json
import sys

# Test the soliton init endpoint
def test_soliton_init():
    url = "http://localhost:8002/api/soliton/init"
    payload = {
        "userId": "adminuser",
        "lattice_reset": False
    }
    
    print(f"Testing POST {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    print("-" * 50)
    
    try:
        response = requests.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {dict(response.headers)}")
        print(f"Response Body:")
        
        try:
            print(json.dumps(response.json(), indent=2))
        except:
            print(response.text)
            
        if response.status_code == 500:
            print("\n❌ Server returned 500 error - check server logs for stack trace")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server at localhost:8002")
        print("Make sure the server is running: poetry run uvicorn prajna_api:app --reload --port 8002")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
        
    return response.status_code == 200

# Test health endpoint first
def test_health():
    url = "http://localhost:8002/api/health"
    print(f"Testing GET {url}")
    print("-" * 50)
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data.get('status', 'unknown')}")
            print(f"Features: {data.get('features', [])}")
            return True
    except:
        print("❌ Health check failed")
        return False
        
    return False

# Check if soliton router is included
def check_openapi():
    url = "http://localhost:8002/openapi.json"
    print(f"\nChecking OpenAPI spec at {url}")
    print("-" * 50)
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            spec = response.json()
            paths = spec.get('paths', {})
            
            # Look for soliton endpoints
            soliton_paths = [p for p in paths if '/soliton' in p]
            
            if soliton_paths:
                print(f"✅ Found {len(soliton_paths)} soliton endpoints:")
                for path in soliton_paths:
                    print(f"  - {path}")
            else:
                print("❌ No soliton endpoints found in API spec!")
                print("The soliton router may not be properly included.")
                
            # Check if /api/soliton/init exists
            if '/api/soliton/init' in paths:
                init_spec = paths['/api/soliton/init']
                print(f"\n/api/soliton/init spec:")
                print(json.dumps(init_spec, indent=2))
            
            return len(soliton_paths) > 0
    except Exception as e:
        print(f"❌ Could not fetch OpenAPI spec: {e}")
        return False

def main():
    print("🔍 Soliton API Diagnostic Tool")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("\n⚠️  API server may not be running properly")
        sys.exit(1)
        
    # Check OpenAPI
    print()
    check_openapi()
    
    # Test soliton init
    print()
    test_soliton_init()
    
    print("\n" + "=" * 50)
    print("Diagnostic complete. Check server logs for detailed error messages.")
    print("\nTo view server logs, run:")
    print("  poetry run uvicorn prajna_api:app --reload --port 8002 --log-level debug")

if __name__ == "__main__":
    main()
