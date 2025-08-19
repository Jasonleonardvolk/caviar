#!/usr/bin/env python
"""
Test script to verify ConceptMesh Rust extension is loading correctly
Run with: python test_conceptmesh_fix.py
"""

import sys
import os
import logging

# Set up logging to see all messages
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

print("=" * 60)
print("CONCEPTMESH RUST EXTENSION TEST")
print("=" * 60)

# Test 1: Direct import of concept_mesh_rs
print("\n1. Testing direct import of concept_mesh_rs...")
try:
    import concept_mesh_rs
    print("✅ SUCCESS: concept_mesh_rs imported!")
    print(f"   Module location: {concept_mesh_rs.__file__}")
    
    # Check version
    if hasattr(concept_mesh_rs, '__version__'):
        print(f"   Version: {concept_mesh_rs.__version__}")
    else:
        print("   Version: Not available")
    
    # Check available attributes
    attrs = [attr for attr in dir(concept_mesh_rs) if not attr.startswith('_')]
    print(f"   Available attributes: {attrs}")
    
except ImportError as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 2: Import ConceptMesh class
print("\n2. Testing ConceptMesh class import...")
try:
    from concept_mesh_rs import ConceptMesh
    print("✅ SUCCESS: ConceptMesh class imported!")
    
    # Check signature (note: PyO3 bindings may not expose readable signatures)
    import inspect
    try:
        sig = inspect.signature(ConceptMesh)
        print(f"   Constructor signature: {sig}")
    except ValueError:
        print("   Constructor signature: Not available (PyO3 binding)")
        print("   This is normal for Rust extensions")
    
except ImportError as e:
    print(f"❌ FAILED: {e}")
    sys.exit(1)

# Test 3: Instantiate ConceptMesh
print("\n3. Testing ConceptMesh instantiation...")
try:
    mesh = ConceptMesh("http://localhost:8003/api/mesh")
    print("✅ SUCCESS: ConceptMesh instantiated!")
    print(f"   Object: {mesh}")
    print(f"   Type: {type(mesh)}")
    
    # Check available methods
    methods = [m for m in dir(mesh) if not m.startswith('_') and callable(getattr(mesh, m))]
    print(f"   Available methods: {methods}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    print(f"   Error type: {type(e).__name__}")

# Test 4: Test soliton_memory import
print("\n4. Testing soliton_memory.py import...")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'mcp_metacognitive', 'core'))

try:
    # Clear any cached imports
    if 'soliton_memory' in sys.modules:
        del sys.modules['soliton_memory']
    
    import soliton_memory
    print("✅ SUCCESS: soliton_memory imported without errors!")
    
    # Check if it loaded the real extension
    if soliton_memory.CONCEPT_MESH_AVAILABLE:
        print("   ✅ Real concept_mesh_rs extension is loaded")
        if soliton_memory.USING_RUST_WHEEL:
            print("   ✅ Using Rust wheel confirmed")
    else:
        print("   ⚠️  Using fallback mock implementation")
    
    # Check if types are available
    print("\n   Checking Python-only types:")
    print(f"   MemoryEntry: {'✅' if hasattr(soliton_memory, 'MemoryEntry') else '❌'}")
    print(f"   MemoryQuery: {'✅' if hasattr(soliton_memory, 'MemoryQuery') else '❌'}")
    print(f"   PhaseTag: {'✅' if hasattr(soliton_memory, 'PhaseTag') else '❌'}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test client initialization
print("\n5. Testing SolitonMemoryClient initialization...")
try:
    client = soliton_memory.SolitonMemoryClient()
    print("✅ SUCCESS: SolitonMemoryClient created!")
    
    if hasattr(client, 'mesh_available'):
        if client.mesh_available:
            print("   ✅ ConceptMesh is available in client")
        else:
            print("   ⚠️  ConceptMesh not available in client")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

# Summary
print("\n📊 SUMMARY:")
print("If all tests passed, you should see:")
print("- 🔧 Loaded real concept_mesh_rs extension")
print("- 📦 concept_mesh_rs version: ...")
print("- ✅ Connected to Concept Mesh")
print("\nWhen you run: poetry run python enhanced_launcher.py")
