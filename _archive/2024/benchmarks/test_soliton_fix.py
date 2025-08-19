"""
Test Soliton Memory API Integration
====================================

Quick test to verify Soliton routes are working.
"""

import asyncio
import aiohttp
import json


async def test_soliton_api():
    """Test Soliton Memory API endpoints"""
    base_url = "http://localhost:8001"  # Main API
    # Or use Prajna URL if testing integrated mode
    # base_url = "http://localhost:8002"  # Prajna
    
    async with aiohttp.ClientSession() as session:
        # Test health check
        print("Testing health check...")
        async with session.get(f"{base_url}/health") as resp:
            print(f"Health: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Test Soliton init
        print("\nTesting Soliton init...")
        payload = {
            "userId": "test_user",
            "config": {"test": True}
        }
        async with session.post(f"{base_url}/api/soliton/init", json=payload) as resp:
            print(f"Init: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Test memory store
        print("\nTesting memory store...")
        payload = {
            "userId": "test_user",
            "content": "This is a test memory about morphons and topology",
            "strength": 0.9
        }
        async with session.post(f"{base_url}/api/soliton/store", json=payload) as resp:
            print(f"Store: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Test memory stats
        print("\nTesting memory stats...")
        async with session.get(f"{base_url}/api/soliton/stats/test_user") as resp:
            print(f"Stats: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Test find memories
        print("\nTesting find memories...")
        payload = {
            "userId": "test_user",
            "query": "morphon",
            "limit": 5
        }
        async with session.post(f"{base_url}/api/soliton/find", json=payload) as resp:
            print(f"Find: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))


async def test_concept_mesh_api():
    """Test Concept Mesh API endpoints"""
    base_url = "http://localhost:8001"
    
    async with aiohttp.ClientSession() as session:
        # Test record diff
        print("\nTesting concept mesh record_diff...")
        payload = {
            "id": "test_concept_1",
            "name": "Test Morphon Concept",
            "embedding": [0.1] * 768,
            "strength": 0.95,
            "metadata": {"source": "test"}
        }
        async with session.post(f"{base_url}/api/concept-mesh/record_diff", json=payload) as resp:
            print(f"Record: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))
        
        # Test stats
        print("\nTesting concept mesh stats...")
        async with session.get(f"{base_url}/api/concept-mesh/stats") as resp:
            print(f"Stats: {resp.status}")
            data = await resp.json()
            print(json.dumps(data, indent=2))


if __name__ == "__main__":
    print("Testing Soliton Memory API Integration")
    print("=" * 40)
    print("\nMake sure to start the API server first:")
    print("  python launch_main_api.py")
    print("\nOr for Prajna integrated mode:")
    print("  python prajna/prajna_api.py")
    print("\n" + "=" * 40 + "\n")
    
    # Run tests
    asyncio.run(test_soliton_api())
    asyncio.run(test_concept_mesh_api())
    
    print("\n✓ Tests complete!")
