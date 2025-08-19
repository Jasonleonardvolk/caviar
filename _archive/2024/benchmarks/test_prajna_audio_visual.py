#!/usr/bin/env python3
"""
Test script for Prajna audio/visual integration
"""

import asyncio
import json
import websockets
import requests
import time
from typing import Dict, Any

# Configuration
PRAJNA_API_BASE = "http://localhost:8001"
WS_ENDPOINT = "ws://localhost:8001/api/stream"

async def test_websocket_connection():
    """Test WebSocket connectivity"""
    print("🔌 Testing WebSocket connection...")
    try:
        async with websockets.connect(WS_ENDPOINT) as websocket:
            print("✅ WebSocket connected successfully")
            
            # Send a test message
            test_request = {
                "user_query": "Hello Prajna, can you hear me?",
                "conversation_id": f"test_{int(time.time())}",
                "streaming": True
            }
            
            await websocket.send(json.dumps(test_request))
            print("📤 Sent test message")
            
            # Listen for response
            response_chunks = []
            async for message in websocket:
                data = json.loads(message)
                print(f"📥 Received: {data.get('type', 'unknown')}")
                
                if data.get('type') == 'chunk':
                    response_chunks.append(data.get('content', ''))
                elif data.get('type') == 'complete':
                    print(f"✅ Complete response: {''.join(response_chunks)}")
                    break
                elif data.get('type') == 'error':
                    print(f"❌ Error: {data.get('error')}")
                    break
                    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

def test_regular_api():
    """Test regular HTTP API"""
    print("\n📡 Testing regular API...")
    
    try:
        # Test health endpoint
        health_response = requests.get(f"{PRAJNA_API_BASE}/health")
        if health_response.status_code == 200:
            print("✅ Health check passed")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
        
        # Test answer endpoint
        test_request = {
            "user_query": "What is TORI?",
            "conversation_id": f"test_{int(time.time())}",
            "streaming": False
        }
        
        response = requests.post(
            f"{PRAJNA_API_BASE}/api/answer",
            json=test_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Got answer: {data.get('answer', '')[:100]}...")
            print(f"   Trust score: {data.get('audit', {}).get('trust_score', 0)}")
        else:
            print(f"❌ Answer request failed: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Prajna API. Is it running?")
    except Exception as e:
        print(f"❌ API error: {e}")

def test_audio_endpoint():
    """Test audio upload endpoint"""
    print("\n🎤 Testing audio endpoint...")
    
    try:
        # Create a dummy audio file (in real case, this would be actual audio)
        import io
        audio_data = io.BytesIO(b"dummy audio data")
        
        files = {
            'audio': ('test.webm', audio_data, 'audio/webm')
        }
        data = {
            'conversation_id': f"test_{int(time.time())}"
        }
        
        response = requests.post(
            f"{PRAJNA_API_BASE}/api/answer/audio",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Audio processed successfully")
            print(f"   Transcription: {result.get('context_used', 'N/A')}")
            print(f"   Answer: {result.get('answer', '')[:100]}...")
        else:
            print(f"❌ Audio processing failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Audio endpoint error: {e}")

async def main():
    """Run all tests"""
    print("🧪 Testing Prajna Audio/Visual Integration\n")
    
    # Test regular API first
    test_regular_api()
    
    # Test WebSocket streaming
    await test_websocket_connection()
    
    # Test audio endpoint
    test_audio_endpoint()
    
    print("\n✨ Testing complete!")

if __name__ == "__main__":
    asyncio.run(main())
