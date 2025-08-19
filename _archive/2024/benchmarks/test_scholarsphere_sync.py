#!/usr/bin/env python3
"""Test ScholarSphere Integration"""

import asyncio
import aiohttp
import json

async def test_scholarsphere():
    """Test the ScholarSphere sync endpoint"""
    
    async with aiohttp.ClientSession() as session:
        # First check if we have concepts
        async with session.get('http://localhost:8002/api/concept_mesh/stats') as resp:
            stats = await resp.json()
            print(f"Current concepts: {stats.get('totalConcepts', 0)}")
        
        # Try to sync
        async with session.post('http://localhost:8002/api/concept_mesh/sync_to_scholarsphere') as resp:
            result = await resp.json()
            print(f"\nSync result: {json.dumps(result, indent=2)}")
            
            if result.get('success'):
                print("\n✅ ScholarSphere sync successful!")
                print(f"   Diff ID: {result.get('diffId')}")
                print(f"   Concepts: {result.get('conceptCount')}")
            else:
                print("\n❌ ScholarSphere sync failed")
                print(f"   Message: {result.get('message')}")

if __name__ == "__main__":
    asyncio.run(test_scholarsphere())
