"""
TORI Chat API Test Script
Test the new chat functionality to ensure everything is working
"""

import requests
import json
import time

def test_chat_api():
    """Test the chat API functionality"""
    
    # Get API URL
    try:
        with open('api_port.json', 'r') as f:
            config = json.load(f)
            api_url = f"http://localhost:{config['port']}"
    except:
        api_url = "http://localhost:8002"
    
    print(f"🤖 Testing TORI Chat API at {api_url}")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{api_url}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✅ Health: {health['status']}")
            print(f"   🔧 Features: {', '.join(health.get('features', []))}")
            print(f"   🤖 Chat enabled: {health.get('chat_enabled', False)}")
            print(f"   🌊 Soliton enabled: {health.get('soliton_enabled', False)}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Chat functionality
    test_messages = [
        "what do you know about darwin",
        "tell me about AI",
        "explain machine learning",
        "what is quantum physics"
    ]
    
    print("\n2. Testing chat messages...")
    for i, message in enumerate(test_messages, 1):
        print(f"\n   Test {i}: '{message}'")
        try:
            chat_response = requests.post(f"{api_url}/chat", json={
                "message": message,
                "userId": "test_user"
            })
            
            if chat_response.status_code == 200:
                result = chat_response.json()
                print(f"   ✅ Response received")
                print(f"   📝 Message: {result['response'][:100]}...")
                print(f"   🧠 Concepts found: {len(result.get('concepts_found', []))}")
                print(f"   📊 Confidence: {result.get('confidence', 0):.2f}")
                print(f"   ⏱️ Processing time: {result.get('processing_time', 0):.2f}s")
                print(f"   🌊 Soliton used: {result.get('soliton_memory_used', False)}")
                
                if result.get('concepts_found'):
                    print(f"   🎯 Found concepts: {', '.join(result['concepts_found'])}")
            else:
                print(f"   ❌ Chat request failed: {chat_response.status_code}")
                print(f"   📄 Response: {chat_response.text}")
        except Exception as e:
            print(f"   ❌ Chat error: {e}")
        
        time.sleep(1)  # Brief pause between tests
    
    print("\n3. Testing frontend API route...")
    try:
        # Test the SvelteKit API route
        frontend_response = requests.post("http://localhost:5173/api/chat", json={
            "message": "test frontend integration",
            "userId": "test_user"
        })
        
        if frontend_response.status_code == 200:
            print("   ✅ Frontend API route working")
        else:
            print(f"   ❌ Frontend API route failed: {frontend_response.status_code}")
    except Exception as e:
        print(f"   ⚠️ Frontend test skipped (not running): {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Chat API testing complete!")
    print("\nTo test in browser:")
    print("1. Go to http://localhost:5173")
    print("2. Use the chat interface")
    print("3. Ask: 'what do you know about darwin'")
    print("4. You should see real concept search results!")

if __name__ == "__main__":
    test_chat_api()
