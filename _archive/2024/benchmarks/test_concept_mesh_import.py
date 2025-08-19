#!/usr/bin/env python3
"""
Test concept_mesh_rs import without Unicode issues
"""

import sys
import subprocess

print("Testing concept_mesh_rs import...")
print("-" * 40)

# Test with proper encoding
test_code = """
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

try:
    import concept_mesh_rs
    print('SUCCESS: concept_mesh_rs imported from:', concept_mesh_rs.__file__)
    loader = concept_mesh_rs.get_loader()
    print('SUCCESS: Loader created:', type(loader).__name__)
except ImportError as e:
    print('FAILED: Import error:', e)
    sys.exit(1)
"""

result = subprocess.run(
    [sys.executable, "-c", test_code],
    capture_output=True,
    text=True,
    encoding='utf-8',
    errors='replace'
)

print(result.stdout)
if result.stderr:
    print("Stderr:", result.stderr)

if result.returncode == 0:
    print("\n✅ concept_mesh_rs is working!")
    
    # Also test in a subprocess (like MCP would)
    print("\nTesting in subprocess (like MCP)...")
    
    subprocess_test = subprocess.run(
        [sys.executable, "-c", "import concept_mesh_rs; print('Subprocess import: OK')"],
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    print(subprocess_test.stdout)
    
    print("\n" + "=" * 60)
    print("✅ SUCCESS! Concept Mesh is built and working!")
    print("\n🚀 Ready to start the server:")
    print("   1. Kill existing processes: taskkill /IM python.exe /F")
    print("   2. Start server: python enhanced_launcher.py")
    print("\n✅ Expected results:")
    print("   - Main process: '🦀 Penrose backend: rust'")
    print("   - MCP subprocess: No 'mock' warnings")
    print("   - Both processes use the same concept_mesh_rs.pyd")
else:
    print("\n❌ Import test failed")
    print("Try manually:")
    print(f"   {sys.executable}")
    print("   >>> import concept_mesh_rs")
