"""
Test script for Phase 1: FractalSolitonMemory Integration
Tests the production soliton API endpoints
"""

import asyncio
import aiohttp
import json
import numpy as np
import time
from typing import Dict, Any

# Test configuration
API_BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_phase1"

async def test_soliton_init():
    """Test soliton initialization"""
    print("\n🧪 Testing Soliton Initialization...")
    
    async with aiohttp.ClientSession() as session:
        # Initialize with some test concepts
        init_data = {
            "user_id": TEST_USER_ID,
            "initial_concepts": [
                {"id": "concept_1", "name": "quantum_memory", "type": "fundamental"},
                {"id": "concept_2", "name": "wave_packet", "type": "physics"}
            ]
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/soliton/init",
            json=init_data
        ) as response:
            result = await response.json()
            print(f"✅ Init Response: {json.dumps(result, indent=2)}")
            assert result["success"] == True
            assert result["engine"] == "fractal_soliton"
            return result

async def test_soliton_store():
    """Test storing memory in soliton lattice"""
    print("\n🧪 Testing Soliton Memory Storage...")
    
    async with aiohttp.ClientSession() as session:
        # Generate a test embedding
        test_embedding = np.random.randn(512).tolist()
        
        store_data = {
            "user_id": TEST_USER_ID,
            "concept_id": f"test_concept_{int(time.time())}",
            "content": {
                "text": "This is a test memory about quantum computing",
                "tags": ["quantum", "computing", "test"],
                "importance": 0.8
            },
            "activation_strength": 0.9,
            "embedding": test_embedding
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/soliton/store",
            json=store_data
        ) as response:
            result = await response.json()
            print(f"✅ Store Response: {json.dumps(result, indent=2)}")
            assert result["success"] == True
            assert "waveProperties" in result
            assert "proofId" in result  # May be None if proof system not available
            return result

async def test_soliton_query():
    """Test querying memories from soliton lattice"""
    print("\n🧪 Testing Soliton Memory Query...")
    
    async with aiohttp.ClientSession() as session:
        # Create a query embedding (similar to stored memory)
        query_embedding = np.random.randn(512).tolist()
        
        query_data = {
            "user_id": TEST_USER_ID,
            "query_embedding": query_embedding,
            "k": 5
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/soliton/query",
            json=query_data
        ) as response:
            result = await response.json()
            print(f"✅ Query Response: {json.dumps(result, indent=2)}")
            assert result["success"] == True
            assert "memories" in result
            return result

async def test_soliton_stats():
    """Test getting soliton statistics"""
    print("\n🧪 Testing Soliton Stats...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/soliton/stats/{TEST_USER_ID}"
        ) as response:
            result = await response.json()
            print(f"✅ Stats Response: {json.dumps(result, indent=2)}")
            assert result["status"] == "operational"
            assert result["totalMemories"] >= 2  # From init
            assert "fieldEnergy" in result
            return result

async def test_soliton_health():
    """Test soliton health endpoint"""
    print("\n🧪 Testing Soliton Health...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/soliton/health"
        ) as response:
            result = await response.json()
            print(f"✅ Health Response: {json.dumps(result, indent=2)}")
            assert result["status"] == "operational"
            assert result["engine"] == "fractal_soliton"
            assert result["features"]["penrose_acceleration"] == True
            return result

async def test_soliton_diagnostic():
    """Test soliton diagnostic endpoint"""
    print("\n🧪 Testing Soliton Diagnostic...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{API_BASE_URL}/api/soliton/diagnostic"
        ) as response:
            result = await response.json()
            print(f"✅ Diagnostic Response: {json.dumps(result, indent=2)}")
            assert result["fractal_soliton_available"] == True
            assert TEST_USER_ID in result["activeInstances"]
            return result

async def run_all_tests():
    """Run all Phase 1 tests"""
    print("🚀 Starting Phase 1 Soliton Integration Tests...")
    print("=" * 60)
    
    try:
        # Run tests in sequence
        await test_soliton_init()
        await asyncio.sleep(1)  # Let lattice initialize
        
        await test_soliton_store()
        await asyncio.sleep(0.5)
        
        await test_soliton_query()
        await test_soliton_stats()
        await test_soliton_health()
        await test_soliton_diagnostic()
        
        print("\n✅ All Phase 1 tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ Test assertion failed: {e}")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_all_tests())
