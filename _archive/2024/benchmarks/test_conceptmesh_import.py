#!/usr/bin/env python3
"""
Quick ConceptMesh diagnostic - run this to verify everything is working
"""

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ensure project root is in path
project_root = Path(r"{PROJECT_ROOT}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("🧪 ConceptMesh Import Test")
print("=" * 40)

# Test 1: Check if the file exists
concept_mesh_file = project_root / "python" / "core" / "concept_mesh.py"
print(f"📁 Checking if file exists: {concept_mesh_file}")
if concept_mesh_file.exists():
    print("   ✅ File exists!")
else:
    print("   ❌ File NOT FOUND!")
    sys.exit(1)

# Test 2: Check python.core.__init__.py
init_file = project_root / "python" / "core" / "__init__.py"
print(f"\n📁 Checking __init__.py: {init_file}")
if init_file.exists():
    print("   ✅ __init__.py exists!")
    # Check if it exports ConceptMesh
    with open(init_file, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'ConceptMesh' in content:
            print("   ✅ ConceptMesh is mentioned in __init__.py")
        else:
            print("   ❌ ConceptMesh NOT mentioned in __init__.py!")
else:
    print("   ❌ __init__.py NOT FOUND!")

# Test 3: Try the import
print("\n🔬 Testing import:")
try:
    from python.core import ConceptMesh
    print("   ✅ SUCCESS: from python.core import ConceptMesh")
    print(f"   📍 Class location: {ConceptMesh.__module__}")
    print(f"   🎯 Class object: {ConceptMesh}")
    
    # Try to instantiate
    try:
        mesh = ConceptMesh()
        print("   ✅ Can instantiate ConceptMesh()")
    except Exception as e:
        print(f"   ⚠️  Instantiation needs parameters: {e}")
        
except ImportError as e:
    print(f"   ❌ IMPORT FAILED: {e}")
    print("\n   💡 Debugging info:")
    
    # Try direct import
    try:
        from python.core.concept_mesh import ConceptMesh as CM2
        print("   ✅ Direct import works: from python.core.concept_mesh import ConceptMesh")
    except ImportError as e2:
        print(f"   ❌ Direct import also failed: {e2}")

# Test 4: Check sys.path
print("\n📂 Python path (first 5 entries):")
for i, path in enumerate(sys.path[:5]):
    print(f"   {i}: {path}")

print("\n" + "=" * 40)
print("🏁 Diagnostic complete!")
print("\n💡 If import failed, check:")
print("   1. Is there a syntax error in concept_mesh.py?")
print("   2. Does concept_mesh.py have missing dependencies?")
print("   3. Is __init__.py properly exporting ConceptMesh?")
