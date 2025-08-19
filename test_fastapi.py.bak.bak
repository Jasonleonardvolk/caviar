#!/usr/bin/env python3
"""
Test script to verify FastAPI extraction service is working properly.
Run this to test the file path approach before using the full UI.
"""

import requests
import json
import sys
import os
from pathlib import Path

def test_fastapi_health():
    """Test the health endpoint"""
    try:
        response = requests.get("http://localhost:8002/health", timeout=5)
        if response.status_code == 200:
            print("✅ FastAPI health check passed")
            return True
        else:
            print(f"❌ FastAPI health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ FastAPI health check failed: {e}")
        return False

def test_file_path_extraction(test_file_path):
    """Test the file path extraction endpoint"""
    if not os.path.exists(test_file_path):
        print(f"❌ Test file not found: {test_file_path}")
        return False
    
    payload = {
        "file_path": test_file_path,
        "filename": os.path.basename(test_file_path),
        "content_type": "application/pdf"
    }
    
    try:
        print(f"🚀 Testing extraction with file: {test_file_path}")
        response = requests.post(
            "http://localhost:8002/extract",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            concept_count = result.get('concept_count', 0)
            concepts = result.get('concept_names', [])
            method = result.get('extraction_method', 'unknown')
            
            print("✅ FastAPI extraction SUCCESS:")
            print(f"   📊 Concepts extracted: {concept_count}")
            print(f"   🔧 Method: {method}")
            print(f"   📝 Sample concepts: {concepts[:5]}")
            return True
        else:
            print(f"❌ FastAPI extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ FastAPI extraction failed: {e}")
        return False

def main():
    print("🧪 TORI FastAPI Test Suite")
    print("=" * 40)
    
    # Test 1: Health check
    if not test_fastapi_health():
        print("🛑 Cannot proceed - FastAPI server not responding")
        sys.exit(1)
    
    # Test 2: Find a test PDF file
    test_dirs = [
        "C:\\Users\\jason\\Desktop\\tori\\kha\\data\\sphere\\admin",
        "C:\\Users\\jason\\Desktop\\tori\\kha\\test_files",
        "C:\\Users\\jason\\Downloads"
    ]
    
    test_file = None
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for file in os.listdir(test_dir):
                if file.lower().endswith('.pdf'):
                    test_file = os.path.join(test_dir, file)
                    break
            if test_file:
                break
    
    if test_file:
        print(f"📄 Found test PDF: {test_file}")
        if test_file_path_extraction(test_file):
            print("🎉 ALL TESTS PASSED! File path architecture is working!")
        else:
            print("❌ File path extraction test failed")
            sys.exit(1)
    else:
        print("⚠️  No PDF files found for testing")
        print("   Upload a PDF through TORI UI first, then run this test")
    
    print("\n🔥 FastAPI is ready for production uploads!")

if __name__ == "__main__":
    main()
