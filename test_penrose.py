#!/usr/bin/env python3
"""
Quick test to check if penrose_engine_rs is available
"""

print("Testing penrose_engine_rs import...")

try:
    import penrose_engine_rs
    print("✅ penrose_engine_rs imported successfully!")
    print(f"Module: {penrose_engine_rs}")
    print(f"Dir: {dir(penrose_engine_rs)}")
except ImportError as e:
    print(f"❌ Failed to import penrose_engine_rs: {e}")

print("\nTesting concept_mesh.similarity import...")
try:
    from concept_mesh import similarity
    print(f"✅ concept_mesh.similarity imported successfully!")
    print(f"Backend: {similarity.BACKEND}")
    print(f"Penrose: {similarity.penrose}")
except ImportError as e:
    print(f"❌ Failed to import concept_mesh.similarity: {e}")
