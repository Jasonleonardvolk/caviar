#!/usr/bin/env python3
"""
Quick Backend Endpoint Test Script
Tests the fixed CORS and soliton endpoints
"""

import requests
import json
import time

def test_endpoint(method, url, data=None, timeout=5):
    """Test an endpoint and return results"""
    try:
        if method.upper() == 'GET':
            response = requests.get(url, timeout=timeout)
        elif method.upper() == 'POST':
            response = requests.post(url, json=data, timeout=timeout)
        
        return {
            "status": "SUCCESS",
            "status_code": response.status_code,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200]
        }
    except requests.exceptions.ConnectionError:
        return {"status": "CONNECTION_ERROR", "message": "Backend not running on port 8002"}
    except requests.exceptions.Timeout:
        return {"status": "TIMEOUT", "message": "Request timed out"}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

def main():
    print("🔬 Testing TORI Backend Endpoints...")
    print("=" * 50)
    
    base_url = "http://localhost:8002"
    
    # Test cases
    tests = [
        ("GET", f"{base_url}/api/health", None, "Basic health check"),
        ("GET", f"{base_url}/api/soliton/health", None, "Soliton engine health"),
        ("POST", f"{base_url}/api/soliton/init", {"userId": "test_user"}, "Soliton initialization"),
        ("POST", f"{base_url}/api/soliton/store", {
            "userId": "test_user", 
            "conceptId": "test_concept", 
            "content": "test memory",
            "importance": 0.8
        }, "Soliton memory store"),
        ("GET", f"{base_url}/api/soliton/recall/test_user/test_concept", None, "Soliton memory recall"),
    ]
    
    results = []
    for method, url, data, description in tests:
        print(f"\n🧪 Testing: {description}")
        print(f"   {method} {url}")
        
        result = test_endpoint(method, url, data)
        results.append((description, result))
        
        if result["status"] == "SUCCESS":
            print(f"   ✅ SUCCESS (HTTP {result['status_code']})")
            if result["status_code"] != 200:
                print(f"   ⚠️  Warning: Non-200 status code")
        else:
            print(f"   ❌ {result['status']}: {result.get('message', 'Unknown error')}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    
    successful = sum(1 for _, result in results if result["status"] == "SUCCESS" and result.get("status_code") == 200)
    total = len(results)
    
    print(f"   ✅ {successful}/{total} endpoints working correctly")
    
    if successful == total:
        print("   🎉 All endpoints are working! CORS and soliton fixes successful.")
    elif successful == 0:
        print("   🚨 Backend appears to be down or not accessible on port 8002")
        print("   💡 Try running: python enhanced_launcher_improved.py")
    else:
        print("   ⚠️  Some endpoints working, some not - check individual results above")
    
    print("\n🔍 Detailed Results:")
    for description, result in results:
        print(f"   {description}: {result['status']}")
        if result["status"] == "SUCCESS" and "response" in result:
            response_preview = str(result["response"])[:100]
            print(f"      Response preview: {response_preview}...")

if __name__ == "__main__":
    main()
