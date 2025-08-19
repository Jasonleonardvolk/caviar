"""
Test script for TONKA integration with Prajna
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8002"

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/api/health")
    data = response.json()
    print(f"✅ Status: {data['status']}")
    print(f"🧠 Prajna: {data.get('prajna_loaded', False)}")
    print(f"🔧 TONKA: {data.get('tonka_ready', False)} ({data.get('tonka_concepts_loaded', 0)} concepts)")
    print(f"📄 PDF Processing: {data.get('pdf_processing_available', False)}")
    return data

def test_tonka_endpoints():
    """Test TONKA-specific endpoints"""
    print("\n🔧 Testing TONKA endpoints...")
    
    # Test TONKA health
    response = requests.get(f"{BASE_URL}/api/tonka/health")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ TONKA Health: {data['status']}")
        print(f"📊 Concepts loaded: {data.get('concepts_loaded', 0)}")
    else:
        print(f"❌ TONKA Health failed: {response.status_code}")
    
    # Test code generation
    print("\n💻 Testing code generation...")
    response = requests.post(
        f"{BASE_URL}/api/tonka/generate",
        json={
            "task": "create a function to calculate factorial",
            "language": "python"
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            print("✅ Code generation successful!")
            print(f"📄 Source: {data.get('source', 'unknown')}")
            print(f"💻 Code:\n{data.get('code', 'No code generated')}")
        else:
            print(f"❌ Code generation failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ Code generation request failed: {response.status_code}")

def test_chat_endpoint():
    """Test the enhanced chat endpoint"""
    print("\n💬 Testing enhanced chat endpoint...")
    
    # Test code generation through chat
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "user_query": "write a function to generate fibonacci numbers",
            "persona": {"name": "Developer"}
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Chat response received!")
        print(f"🔧 Code generated: {data.get('code_generated', False)}")
        print(f"📄 Answer preview: {data.get('answer', '')[:200]}...")
    else:
        print(f"❌ Chat request failed: {response.status_code}")
    
    # Test non-code query
    print("\n📚 Testing non-code query...")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={
            "user_query": "What is consciousness?",
            "persona": {"name": "Philosopher"}
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Non-code response received!")
        print(f"📄 Sources: {data.get('sources', [])}")
        print(f"📄 Answer preview: {data.get('answer', '')[:200]}...")
    else:
        print(f"❌ Non-code query failed: {response.status_code}")

def test_project_creation():
    """Test project creation"""
    print("\n🏗️ Testing project creation...")
    
    response = requests.post(
        f"{BASE_URL}/api/tonka/project",
        json={
            "project_type": "api",
            "name": "test_api",
            "features": ["file_storage", "auth"]
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        if data.get("success"):
            print("✅ Project creation successful!")
            print(f"📁 Files created: {data.get('file_count', 0)}")
            files = data.get('files', {})
            for filename in list(files.keys())[:3]:
                print(f"  - {filename}")
        else:
            print(f"❌ Project creation failed: {data.get('error', 'Unknown error')}")
    else:
        print(f"❌ Project creation request failed: {response.status_code}")

def main():
    """Run all tests"""
    print("🧪 TONKA Integration Test Suite")
    print("=" * 50)
    
    try:
        # Test health first
        health_data = test_health()
        
        # Only test TONKA if it's available
        if health_data.get('tonka_available'):
            test_tonka_endpoints()
            test_chat_endpoint()
            test_project_creation()
        else:
            print("\n⚠️ TONKA not available - skipping TONKA-specific tests")
        
        print("\n✅ All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server at", BASE_URL)
        print("Make sure the server is running with: python enhanced_launcher.py")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")

if __name__ == "__main__":
    main()
