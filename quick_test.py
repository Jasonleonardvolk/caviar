#!/usr/bin/env python
"""
Quick test to see if ConceptMesh loads correctly
"""
import os
import sys

# Add the path for soliton_memory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_metacognitive', 'core'))

print("Testing ConceptMesh loading...")
print("-" * 40)

# Test 1: Direct module test
try:
    import concept_mesh_rs
    print("✅ concept_mesh_rs module loaded")
    print(f"   Location: {concept_mesh_rs.__file__}")
except Exception as e:
    print(f"❌ Failed to load module: {e}")

# Test 2: ConceptMesh instantiation
try:
    from concept_mesh_rs import ConceptMesh
    mesh = ConceptMesh("http://localhost:8003/api/mesh")
    print("✅ ConceptMesh instantiated successfully!")
except Exception as e:
    print(f"❌ Failed to instantiate: {e}")

# Test 3: Import soliton_memory
try:
    import soliton_memory
    print("\n✅ soliton_memory imported successfully!")
    print(f"   CONCEPT_MESH_AVAILABLE: {soliton_memory.CONCEPT_MESH_AVAILABLE}")
    print(f"   USING_RUST_WHEEL: {soliton_memory.USING_RUST_WHEEL}")
except Exception as e:
    print(f"\n❌ Failed to import soliton_memory: {e}")

print("\n" + "-" * 40)
print("If you see ✅ for all tests, the fix is working!")
