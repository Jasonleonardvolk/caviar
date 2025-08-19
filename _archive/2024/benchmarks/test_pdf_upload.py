#!/usr/bin/env python3
"""
Test PDF Upload to TORI Backend
This script tests if the upload endpoint is working correctly.
"""

import requests
import os
import sys
from pathlib import Path

def test_upload():
    """Test the upload endpoint"""
    
    # Backend URL
    backend_url = "http://localhost:8002"
    
    print("🔍 Testing TORI Backend Upload...")
    print("-" * 50)
    
    # 1. Test health endpoint
    print("\n1. Testing /api/health endpoint...")
    try:
        response = requests.get(f"{backend_url}/api/health", timeout=5)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Health check passed!")
            print(f"   Response: {response.json()}")
        else:
            print(f"   ❌ Health check failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print("   Make sure the backend is running on port 8002!")
        return
    
    # 2. Test upload endpoint
    print("\n2. Testing /upload endpoint...")
    
    # Create a test PDF file
    test_file = "test_upload.txt"
    with open(test_file, "w") as f:
        f.write("This is a test file for TORI upload testing.\n" * 100)
    
    try:
        with open(test_file, "rb") as f:
            files = {"file": ("test_document.pdf", f, "application/pdf")}
            response = requests.post(f"{backend_url}/upload", files=files, timeout=30)
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Upload successful!")
            result = response.json()
            print(f"   File path: {result.get('file_path')}")
            print(f"   Size: {result.get('size_mb')} MB")
            
            # 3. Test extraction
            print("\n3. Testing /extract endpoint...")
            extract_data = {
                "file_path": result.get("file_path"),
                "filename": result.get("filename"),
                "content_type": "application/pdf"
            }
            
            extract_response = requests.post(
                f"{backend_url}/extract", 
                json=extract_data,
                timeout=60
            )
            
            print(f"   Status: {extract_response.status_code}")
            if extract_response.status_code == 200:
                print("   ✅ Extraction successful!")
                extract_result = extract_response.json()
                print(f"   Concepts found: {extract_result.get('concept_count', 0)}")
            else:
                print(f"   ❌ Extraction failed: {extract_response.text}")
                
        else:
            print(f"   ❌ Upload failed: {response.text}")
            
    except requests.exceptions.Timeout:
        print("   ❌ Request timed out - backend might be stuck")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    finally:
        # Clean up test file
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print("\n" + "-" * 50)
    print("Test complete!")
    
    # 4. Check CORS headers
    print("\n4. Checking CORS configuration...")
    try:
        # Simulate browser preflight request
        headers = {
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"
        }
        response = requests.options(f"{backend_url}/upload", headers=headers, timeout=5)
        print(f"   Preflight status: {response.status_code}")
        print(f"   CORS headers: {dict(response.headers)}")
    except Exception as e:
        print(f"   ❌ CORS check error: {e}")

if __name__ == "__main__":
    test_upload()
