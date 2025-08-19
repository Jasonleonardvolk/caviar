#!/usr/bin/env python3
"""
TORI Authentication Test Script
Demonstrates proper Bearer token authentication for PDF upload
"""

import requests
import json
import sys
import os

def test_tori_authentication():
    """Test TORI authentication and upload with proper Bearer token"""
    
    TORI_HOST = "localhost:8443"
    BASE_URL = f"http://{TORI_HOST}"
    
    print("🔐 TORI Authentication & Upload Test")
    print("=" * 40)
    
    # Test connection
    print("🌐 Testing connection to TORI...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ TORI server is responding")
        else:
            print(f"⚠️ TORI responded with status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot reach TORI server: {e}")
        print("   Make sure TORI is running with:")
        print("   python phase3_complete_production_system.py --host 0.0.0.0 --port 8443")
        return False
    
    # Step 1: Login to get token
    print("\n🔐 Step 1: Getting authentication token...")
    login_data = {
        "username": "operator",
        "password": "operator123"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/auth/login",
            headers={"Content-Type": "application/json"},
            json=login_data,
            timeout=10
        )
        
        print(f"📋 Login Response Status: {response.status_code}")
        print(f"📋 Login Response: {response.text}")
        
        if response.status_code == 200:
            login_result = response.json()
            token = login_result.get("token")
            
            if token:
                print(f"✅ Token obtained: {token[:20]}...")
            else:
                print("❌ No token in response")
                return False
        else:
            print("❌ Login failed")
            return False
            
    except Exception as e:
        print(f"❌ Login error: {e}")
        return False
    
    # Step 2: Test PDF upload with Bearer token
    print("\n📤 Step 2: Testing PDF upload with Bearer token...")
    
    # Create a test file if needed
    test_file = "test_document.pdf"
    if not os.path.exists(test_file):
        print(f"📄 Creating test file: {test_file}")
        with open(test_file, "w") as f:
            f.write("This is a test PDF content for TORI authentication testing")
    
    # Prepare headers with Bearer token
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    print(f"🔑 Using Authorization header: Bearer {token[:20]}...")
    
    try:
        with open(test_file, "rb") as f:
            files = {"file": (test_file, f, "application/pdf")}
            
            response = requests.post(
                f"{BASE_URL}/api/upload",
                headers=headers,
                files=files,
                timeout=30
            )
        
        print(f"📋 Upload Response Status: {response.status_code}")
        print(f"📋 Upload Response: {response.text}")
        
        if response.status_code == 200:
            print("🎆 SUCCESS! PDF uploaded successfully!")
            
            # Try to parse response for extraction
            try:
                upload_result = response.json()
                file_path = upload_result.get("file_path")
                
                if file_path:
                    print(f"\n🧬 Step 3: Testing concept extraction...")
                    extract_data = {"file_path": file_path}
                    
                    extract_response = requests.post(
                        f"{BASE_URL}/api/extract",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {token}"
                        },
                        json=extract_data,
                        timeout=30
                    )
                    
                    print(f"📊 Extraction Response Status: {extract_response.status_code}")
                    print(f"📊 Extraction Response: {extract_response.text}")
                    
                    if extract_response.status_code == 200:
                        print("🎆 COMPLETE SUCCESS! Concepts extracted!")
                    else:
                        print("⚠️ Extraction failed, but upload succeeded")
                        
            except Exception as e:
                print(f"⚠️ Could not parse upload response: {e}")
                
        elif response.status_code == 403:
            print("❌ 403 Forbidden - Authorization failed")
            print("   Check that Bearer token is correct")
        elif response.status_code == 401:
            print("❌ 401 Unauthorized - Token may be invalid or expired")
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return False
    
    print("\n🎯 Authentication test completed!")
    
    # Show manual cURL commands
    print("\n💡 Manual cURL commands:")
    print(f"1. Login:")
    print(f'   curl -X POST "{BASE_URL}/api/auth/login" \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f"     -d '{{\"username\":\"operator\",\"password\":\"operator123\"}}'")
    print(f"\n2. Upload:")
    print(f'   curl -X POST "{BASE_URL}/api/upload" \\')
    print(f'     -H "Authorization: Bearer {token}" \\')
    print(f'     -F "file=@{test_file};type=application/pdf"')
    
    return True

if __name__ == "__main__":
    success = test_tori_authentication()
    sys.exit(0 if success else 1)
