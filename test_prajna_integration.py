#!/usr/bin/env python3
"""
Test script to validate Prajna mouth integration with API
========================================================

This script tests that the Prajna API integration is working correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the parent directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

async def test_prajna_integration():
    """Test Prajna language model integration"""
    try:
        print("🧪 Testing Prajna Integration...")
        print("=" * 50)
        
        # Test 1: Import the mouth module
        print("📦 Testing imports...")
        from prajna.core.prajna_mouth import PrajnaLanguageModel, generate_prajna_response, PrajnaOutput
        print("✅ Successfully imported Prajna mouth components")
        
        # Test 2: Create and load model
        print("\n🗣️ Testing model creation and loading...")
        model = PrajnaLanguageModel(model_type="demo", temperature=0.7)
        await model.load_model()
        print(f"✅ Model loaded: {model.model_type}")
        print(f"✅ Model ready: {model.is_loaded()}")
        
        # Test 3: Generate response
        print("\n💬 Testing response generation...")
        test_query = "What is Prajna?"
        response = await model.generate_response(test_query)
        
        print(f"📝 Query: {test_query}")
        print(f"🗣️ Response: {response.answer[:100]}...")
        print(f"⏱️ Processing time: {response.processing_time:.3f}s")
        print(f"🎯 Confidence: {response.confidence}")
        print(f"🔧 Model used: {response.model_used}")
        print(f"📊 Tokens generated: {response.tokens_generated}")
        
        # Test 4: Get model stats
        print("\n📊 Testing model statistics...")
        stats = await model.get_stats()
        print(f"✅ Stats retrieved: {stats['total_requests']} total requests")
        print(f"✅ Average response time: {stats['average_response_time']:.3f}s")
        
        # Test 5: Test configuration
        print("\n⚙️ Testing configuration...")
        from prajna.api.prajna_api import PrajnaSettings
        settings = PrajnaSettings()
        print(f"✅ Model type: {settings.model_type}")
        print(f"✅ Temperature: {settings.temperature}")
        print(f"✅ Max context length: {settings.max_context_length}")
        
        print("\n🎉 All tests passed! Prajna integration is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure the prajna package is in your Python path")
        return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

async def test_api_imports():
    """Test API integration imports"""
    try:
        print("\n🔌 Testing API integration...")
        
        # Test API imports
        from prajna.api.prajna_api import app, settings, load_prajna_model, gather_context
        print("✅ API components imported successfully")
        
        # Test configuration
        print(f"✅ Settings loaded: {settings.model_type} model")
        
        # Test context gathering
        context = await gather_context("test_user", "test_conversation")
        print(f"✅ Context gathering works: '{context}'")
        
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 Starting Prajna Integration Tests")
    print("=" * 60)
    
    # Test basic integration
    basic_success = await test_prajna_integration()
    
    # Test API integration
    api_success = await test_api_imports()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY:")
    print(f"   Basic Integration: {'✅ PASS' if basic_success else '❌ FAIL'}")
    print(f"   API Integration:   {'✅ PASS' if api_success else '❌ FAIL'}")
    
    if basic_success and api_success:
        print("\n🎯 ALL TESTS PASSED! Prajna is ready to use.")
        print("\n📋 Next Steps:")
        print("   1. Start the API server: python -m uvicorn prajna.api.prajna_api:app --reload")
        print("   2. Test endpoint: POST http://localhost:8000/api/answer")
        print("   3. Check health: GET http://localhost:8000/api/health")
        print("   4. View stats: GET http://localhost:8000/api/prajna/stats")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("Please check the error messages above and fix any issues.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
