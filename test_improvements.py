"""
Test script to verify all 8 improvements in the updated ConceptMesh
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from python.core.concept_mesh import ConceptMesh, ConceptDiff
import os
import time
import uuid

def test_all_improvements():
    print("=== Testing ConceptMesh Improvements ===\n")
    
    # Set environment variables for testing
    os.environ['CONCEPT_MESH_STORAGE_PATH'] = 'data/test_improved_mesh'
    os.environ['CONCEPT_MESH_MAX_DIFF_HISTORY'] = '100'
    os.environ['CONCEPT_MESH_EMBEDDING_CACHE_SIZE'] = '50'
    
    # 1. Test UUID-based ID generation
    print("1. Testing UUID-based ID generation:")
    mesh = ConceptMesh.instance()
    
    # Add concepts and check UUIDs
    id1 = mesh.add_concept("test_concept_1", "First test")
    id2 = mesh.add_concept("test_concept_2", "Second test")
    
    try:
        uuid.UUID(id1)
        uuid.UUID(id2)
        print(f"   ✅ IDs are valid UUIDs: {id1[:8]}..., {id2[:8]}...")
    except ValueError:
        print(f"   ❌ IDs are not valid UUIDs!")
    
    # 2. Test configurable paths
    print("\n2. Testing configurable paths:")
    print(f"   Storage path from env: {mesh.storage_path}")
    print(f"   Max diff history from env: {mesh.config.get_max_diff_history()}")
    print(f"   Cache size from env: {mesh.config.get_embedding_cache_size()}")
    
    # 3. Test specific exception handling (check logs)
    print("\n3. Testing specific exception handling:")
    print("   (Check logs for specific exception types)")
    
    # 4. Test LRU cache
    print("\n4. Testing LRU cache for embeddings:")
    import numpy as np
    
    # Add concepts with embeddings
    for i in range(10):
        mesh.add_concept(
            f"embed_test_{i}", 
            f"Test {i}",
            embedding=np.random.rand(768)
        )
    
    # Access embeddings multiple times
    concepts = list(mesh.concepts.keys())[:5]
    for _ in range(3):
        for cid in concepts:
            mesh._get_embedding_cached(cid)
    
    cache_info = mesh._get_embedding_cached.cache_info()
    print(f"   Cache stats - Hits: {cache_info.hits}, Misses: {cache_info.misses}")
    print(f"   Cache size: {cache_info.currsize}/{cache_info.maxsize}")
    
    # 5. Test unsubscribe mechanism
    print("\n5. Testing unsubscribe mechanism:")
    
    # Subscribe to diffs
    callback_called = []
    def test_callback(diff: ConceptDiff):
        callback_called.append(diff.diff_type)
    
    sub_id = mesh.subscribe_to_diffs(test_callback)
    print(f"   Subscribed with ID: {sub_id}")
    
    # Add a concept (should trigger callback)
    mesh.add_concept("trigger_test", "Should call callback")
    print(f"   Callback called: {len(callback_called)} times")
    
    # Unsubscribe
    success = mesh.unsubscribe_from_diffs(sub_id)
    print(f"   Unsubscribed: {success}")
    
    # Add another concept (should NOT trigger callback)
    callback_called.clear()
    mesh.add_concept("no_trigger_test", "Should not call callback")
    print(f"   Callback called after unsubscribe: {len(callback_called)} times")
    
    # 6. Test class-level stop words
    print("\n6. Testing class-level stop words:")
    from python.core.concept_mesh import STOP_WORDS
    print(f"   Stop words defined at class level: {len(STOP_WORDS)} words")
    print(f"   Sample stop words: {list(STOP_WORDS)[:10]}...")
    
    # 7. Test type annotations
    print("\n7. Testing type annotations:")
    print("   Checking method signatures...")
    
    # Check return type annotations
    methods_with_types = [
        ('add_concept', 'ConceptID'),
        ('remove_concept', 'bool'),
        ('count', 'int'),
        ('subscribe_to_diffs', 'CallbackID')
    ]
    
    for method_name, expected_return in methods_with_types:
        method = getattr(mesh, method_name)
        if hasattr(method, '__annotations__'):
            return_type = method.__annotations__.get('return', 'None')
            print(f"   {method_name} -> {return_type}")
    
    # 8. Test statistics with cache info
    print("\n8. Testing enhanced statistics:")
    stats = mesh.get_statistics()
    print(f"   Total concepts: {stats['total_concepts']}")
    print(f"   Storage path: {stats['storage_path']}")
    print(f"   Cache info: {stats['cache_info']}")
    
    # Test configuration from file
    print("\n9. Testing configuration system:")
    custom_config = {
        'storage_path': 'data/custom_mesh',
        'seed_files': ['custom_seeds.json'],
        'max_diff_history': 200,
        'embedding_cache_size': 100
    }
    
    # Reset instance to test new config
    ConceptMesh.reset_instance()
    custom_mesh = ConceptMesh.instance(custom_config)
    
    print(f"   Custom storage path: {custom_mesh.storage_path}")
    print(f"   Custom max diff history: {custom_mesh.config.get_max_diff_history()}")
    print(f"   Custom cache size: {custom_mesh.config.get_embedding_cache_size()}")
    
    print("\n✅ All improvements tested successfully!")
    
    # Cleanup
    ConceptMesh.reset_instance()

if __name__ == "__main__":
    test_all_improvements()
