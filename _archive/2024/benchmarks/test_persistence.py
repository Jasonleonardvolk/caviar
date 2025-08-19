"""
Test script to verify ConceptMesh persistence is working correctly.
Run this to see the persistence in action.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from python.core.concept_mesh import ConceptMesh
import time

def test_persistence():
    print("=== ConceptMesh Persistence Test ===\n")
    
    # Get the mesh instance
    mesh = ConceptMesh.instance()
    
    # Show current state
    print(f"1. Initial state: {mesh.count()} concepts")
    
    # Add a test concept with timestamp
    test_name = f"test_concept_{int(time.time())}"
    test_id = mesh.add_concept(
        name=test_name,
        description="A test concept to verify persistence",
        category="test",
        importance=0.5
    )
    
    print(f"2. Added concept: '{test_name}' (ID: {test_id})")
    print(f"   Total concepts now: {mesh.count()}")
    
    # Force save
    mesh._save_mesh()
    print("3. Manually saved mesh to disk")
    
    # Show statistics
    stats = mesh.get_statistics()
    print(f"\n4. Mesh Statistics:")
    print(f"   - Total concepts: {stats['total_concepts']}")
    print(f"   - Total relations: {stats['total_relations']}")
    print(f"   - Categories: {list(stats['categories'].keys())}")
    print(f"   - Storage path: {stats['storage_path']}")
    
    # Show some existing concepts
    print(f"\n5. Sample concepts:")
    for i, (cid, concept) in enumerate(list(mesh.concepts.items())[:5]):
        print(f"   - {concept.name} ({concept.category})")
    
    print(f"\n✅ Test complete! The mesh should persist across restarts.")
    print(f"   Next time you run this, you should see {mesh.count()} or more concepts.")
    print(f"   Look for concept '{test_name}' to verify persistence.")

if __name__ == "__main__":
    test_persistence()
