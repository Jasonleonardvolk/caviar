"""
Test script for Phase 3: Scope Concept Mesh to Users and Tenants
Tests multi-tenancy, WAL isolation, ALBERT physics, and proof contexts
"""

import asyncio
import aiohttp
import json
import time
import numpy as np
from typing import Dict, Any, List

# Test configuration
API_BASE_URL = "http://localhost:8000"

# Test users and groups
USER1 = "alice_phase3"
USER2 = "bob_phase3"
GROUP1 = "research_group_phase3"

async def test_mesh_isolation():
    """Test that user meshes are isolated"""
    print("\n🧪 Testing Mesh Isolation...")
    
    async with aiohttp.ClientSession() as session:
        # Add concept to user1's mesh
        concept1_data = {
            "name": "quantum_computing",
            "description": "Study of quantum computers",
            "category": "technology",
            "importance": 8.5
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER1}",
            json=concept1_data
        ) as response:
            result = await response.json()
            print(f"✅ User1 added concept: {result['name']}")
            assert response.status == 201
        
        # Add different concept to user2's mesh
        concept2_data = {
            "name": "classical_computing",
            "description": "Traditional computing",
            "category": "technology",
            "importance": 7.0
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER2}",
            json=concept2_data
        ) as response:
            result = await response.json()
            print(f"✅ User2 added concept: {result['name']}")
            assert response.status == 201
        
        # Verify user1 doesn't see user2's concept
        async with session.get(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER1}"
        ) as response:
            result = await response.json()
            user1_concepts = [c['name'] for c in result['concepts']]
            assert "quantum_computing" in user1_concepts
            assert "classical_computing" not in user1_concepts
            print(f"✅ User1 concepts isolated: {user1_concepts}")
        
        # Verify user2 doesn't see user1's concept
        async with session.get(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER2}"
        ) as response:
            result = await response.json()
            user2_concepts = [c['name'] for c in result['concepts']]
            assert "classical_computing" in user2_concepts
            assert "quantum_computing" not in user2_concepts
            print(f"✅ User2 concepts isolated: {user2_concepts}")

async def test_etag_changes():
    """Test ETag changes with mesh updates"""
    print("\n🧪 Testing ETag Changes...")
    
    async with aiohttp.ClientSession() as session:
        # Get initial ETag for new group
        async with session.get(
            f"{API_BASE_URL}/api/mesh/etag?groupId={GROUP1}"
        ) as response:
            if response.status == 404:
                print("✅ New group correctly returns 404 (no mesh yet)")
                initial_etag = None
            else:
                result = await response.json()
                initial_etag = result.get('etag')
        
        # Add concept to group
        concept_data = {
            "name": "group_knowledge",
            "description": "Shared group concept",
            "category": "collaboration"
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?groupId={GROUP1}",
            json=concept_data
        ) as response:
            assert response.status == 201
        
        # Get new ETag
        async with session.get(
            f"{API_BASE_URL}/api/mesh/etag?groupId={GROUP1}"
        ) as response:
            result = await response.json()
            new_etag = result['etag']
            assert new_etag != initial_etag
            print(f"✅ ETag changed after update: {initial_etag} → {new_etag}")

async def test_wal_persistence():
    """Test WAL persistence and replay"""
    print("\n🧪 Testing WAL Persistence...")
    
    test_user = f"wal_test_{int(time.time())}"
    
    async with aiohttp.ClientSession() as session:
        # Add multiple concepts
        concepts = ["concept_a", "concept_b", "concept_c"]
        
        for concept_name in concepts:
            async with session.post(
                f"{API_BASE_URL}/api/mesh/concepts?userId={test_user}",
                json={"name": concept_name}
            ) as response:
                assert response.status == 201
        
        # Get WAL stats
        async with session.get(
            f"{API_BASE_URL}/api/mesh/wal/stats?userId={test_user}"
        ) as response:
            result = await response.json()
            wal_stats = result['wal_stats']
            print(f"✅ WAL stats: {json.dumps(wal_stats, indent=2)}")
            assert wal_stats['sequence'] >= 3  # At least 3 operations
        
        # Force checkpoint
        async with session.post(
            f"{API_BASE_URL}/api/mesh/checkpoint?userId={test_user}"
        ) as response:
            result = await response.json()
            print(f"✅ Checkpoint created: {result['message']}")

async def test_albert_physics():
    """Test ALBERT physics computation"""
    print("\n🧪 Testing ALBERT Physics Engine...")
    
    physics_user = "physics_test"
    
    async with aiohttp.ClientSession() as session:
        # Create a small concept network
        concepts = [
            {"name": "energy", "importance": 9.0},
            {"name": "mass", "importance": 8.5},
            {"name": "speed_of_light", "importance": 10.0}
        ]
        
        for concept in concepts:
            async with session.post(
                f"{API_BASE_URL}/api/mesh/concepts?userId={physics_user}",
                json=concept
            ) as response:
                assert response.status == 201
        
        # Add relations (E=mc²)
        relations = [
            {
                "source_name": "energy",
                "target_name": "mass",
                "relation_type": "equals",
                "strength": 1.0
            },
            {
                "source_name": "mass",
                "target_name": "speed_of_light",
                "relation_type": "multiplied_by_squared",
                "strength": 1.0
            }
        ]
        
        for relation in relations:
            async with session.post(
                f"{API_BASE_URL}/api/mesh/relations?userId={physics_user}",
                json=relation
            ) as response:
                assert response.status == 200
        
        # Get physics properties
        async with session.get(
            f"{API_BASE_URL}/api/mesh/physics?userId={physics_user}"
        ) as response:
            result = await response.json()
            print(f"✅ Physics properties: {json.dumps(result, indent=2)}")
            assert 'free_energy' in result
            assert 'health' in result
            assert result['health']['health_score'] > 0

async def test_tenant_proofs():
    """Test tenant-isolated proof generation"""
    print("\n🧪 Testing Tenant-Isolated Proofs...")
    
    async with aiohttp.ClientSession() as session:
        # Add concept with proof generation
        concept_data = {
            "name": "verified_fact",
            "description": "A fact to be formally verified",
            "generate_proof": True
        }
        
        # Add to user mesh
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER1}",
            json=concept_data
        ) as response:
            result = await response.json()
            user_proof_id = result.get('proof_id')
            print(f"✅ User proof generated: {user_proof_id}")
            assert user_proof_id is not None
        
        # Add to group mesh
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?groupId={GROUP1}",
            json=concept_data
        ) as response:
            result = await response.json()
            group_proof_id = result.get('proof_id')
            print(f"✅ Group proof generated: {group_proof_id}")
            assert group_proof_id is not None
        
        # Get proof stats
        await asyncio.sleep(1)  # Let proofs process
        
        async with session.get(
            f"{API_BASE_URL}/api/mesh/proof/stats?userId={USER1}"
        ) as response:
            result = await response.json()
            user_stats = result['proof_stats']
            print(f"✅ User proof stats: {json.dumps(user_stats, indent=2)}")
            assert user_stats['total_proofs'] > 0

async def test_mesh_merge():
    """Test merging meshes"""
    print("\n🧪 Testing Mesh Merge...")
    
    merge_user = "merge_source"
    merge_group = "merge_target_group"
    
    async with aiohttp.ClientSession() as session:
        # Add concepts to user mesh
        user_concepts = ["personal_idea", "private_thought"]
        
        for concept in user_concepts:
            async with session.post(
                f"{API_BASE_URL}/api/mesh/concepts?userId={merge_user}",
                json={"name": concept}
            ) as response:
                assert response.status == 201
        
        # Add concept to group mesh
        async with session.post(
            f"{API_BASE_URL}/api/mesh/concepts?groupId={merge_group}",
            json={"name": "group_concept"}
        ) as response:
            assert response.status == 201
        
        # Merge user mesh into group
        merge_request = {
            "source_scope": "user",
            "source_id": merge_user,
            "merge_strategy": "union"
        }
        
        async with session.post(
            f"{API_BASE_URL}/api/mesh/merge/group/{merge_group}",
            json=merge_request
        ) as response:
            result = await response.json()
            print(f"✅ Merge successful: {result}")
            assert result['success'] == True
        
        # Verify merged concepts
        async with session.get(
            f"{API_BASE_URL}/api/mesh/concepts?groupId={merge_group}"
        ) as response:
            result = await response.json()
            group_concepts = [c['name'] for c in result['concepts']]
            assert "personal_idea" in group_concepts
            assert "private_thought" in group_concepts
            assert "group_concept" in group_concepts
            print(f"✅ Merged concepts: {group_concepts}")

async def test_concurrent_writes():
    """Test concurrent writes to different meshes"""
    print("\n🧪 Testing Concurrent Writes...")
    
    async with aiohttp.ClientSession() as session:
        # Create concurrent write tasks
        tasks = []
        
        # 5 writes to user1
        for i in range(5):
            task = session.post(
                f"{API_BASE_URL}/api/mesh/concepts?userId={USER1}_concurrent",
                json={"name": f"user1_concept_{i}"}
            )
            tasks.append(task)
        
        # 5 writes to user2
        for i in range(5):
            task = session.post(
                f"{API_BASE_URL}/api/mesh/concepts?userId={USER2}_concurrent",
                json={"name": f"user2_concept_{i}"}
            )
            tasks.append(task)
        
        # Execute all concurrently
        responses = await asyncio.gather(*[task.__aenter__() for task in tasks])
        
        # Verify all succeeded
        for response in responses:
            assert response.status == 201
        
        # Clean up
        for response in responses:
            await response.__aexit__(None, None, None)
        
        print("✅ All concurrent writes succeeded")
        
        # Verify isolation
        async with session.get(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER1}_concurrent"
        ) as response:
            result = await response.json()
            assert result['total'] == 5
            
        async with session.get(
            f"{API_BASE_URL}/api/mesh/concepts?userId={USER2}_concurrent"
        ) as response:
            result = await response.json()
            assert result['total'] == 5
            
        print("✅ Concurrent writes remained isolated")

async def run_all_tests():
    """Run all Phase 3 tests"""
    print("🚀 Starting Phase 3 Multi-Tenancy Tests...")
    print("=" * 60)
    
    try:
        # Core isolation tests
        await test_mesh_isolation()
        await test_etag_changes()
        
        # Persistence tests
        await test_wal_persistence()
        
        # Physics tests
        await test_albert_physics()
        
        # Proof tests
        await test_tenant_proofs()
        
        # Advanced features
        await test_mesh_merge()
        await test_concurrent_writes()
        
        print("\n✅ All Phase 3 tests passed!")
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
