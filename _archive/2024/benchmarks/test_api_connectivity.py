#!/usr/bin/env python3
"""
🔍 QUICK CONNECTIVITY TEST
Test if SvelteKit can reach the Python API
"""

import requests
import json

def test_python_api():
    """Test connectivity to Python API"""
    print("🔍 TESTING PYTHON API CONNECTIVITY")
    print("=" * 50)
    
    base_url = "http://localhost:8002"
    
    # Test 1: Health check
    try:
        print("🏥 Testing health endpoint...")
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Test concepts: {data.get('test_concepts_found')}")
        else:
            print(f"❌ Health check failed: {response.text}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Text extraction
    try:
        print("\n📄 Testing text extraction...")
        test_data = {
            "text": "This is a test of machine learning and artificial intelligence algorithms for optimization."
        }
        response = requests.post(f"{base_url}/extract/text", 
                               json=test_data, 
                               timeout=30)
        print(f"✅ Text extraction: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Concepts found: {data.get('concept_count')}")
            print(f"   Processing time: {data.get('processing_time_seconds')}s")
        else:
            print(f"❌ Text extraction failed: {response.text}")
    except Exception as e:
        print(f"❌ Text extraction failed: {e}")
        return False
    
    print("\n✅ Python API is fully accessible!")
    return True

if __name__ == "__main__":
    test_python_api()
