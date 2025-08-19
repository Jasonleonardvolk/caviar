#!/usr/bin/env python3
"""
Diagnose TorusCells import issue
"""
import sys
import traceback

print("=== TorusCells Import Diagnostic ===")
print()

# Step 1: Check if torus_registry imports
print("1. Testing TorusRegistry import...")
try:
    from python.core.torus_registry import TorusRegistry, get_torus_registry
    print("   ✅ TorusRegistry imported successfully")
except Exception as e:
    print(f"   ❌ TorusRegistry import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Check topology libraries
print("\n2. Checking topology libraries...")
try:
    import gudhi
    print("   ✅ GUDHI is available")
except ImportError:
    print("   ⚠️ GUDHI not installed (optional)")

try:
    import ripser
    print("   ✅ Ripser is available")
except ImportError:
    print("   ⚠️ Ripser not installed (optional)")

# Step 3: Check scipy (needed for fallback)
print("\n3. Checking scipy (needed for fallback)...")
try:
    from scipy.spatial.distance import pdist, squareform
    print("   ✅ scipy is available")
except ImportError as e:
    print(f"   ❌ scipy not available: {e}")

# Step 4: Try to import TorusCells
print("\n4. Testing TorusCells import...")
try:
    from python.core.torus_cells import TorusCells, get_torus_cells, betti0_1
    print("   ✅ TorusCells imported successfully!")
    
    # Try to create an instance
    print("\n5. Creating TorusCells instance...")
    cells = get_torus_cells()
    print(f"   ✅ TorusCells instance created with backend: {cells.backend}")
    
    # Test basic functionality
    print("\n6. Testing basic functionality...")
    import numpy as np
    test_vertices = np.random.rand(10, 3)
    b0, b1 = betti0_1(test_vertices)
    print(f"   ✅ Betti numbers computed: b0={b0}, b1={b1}")
    
except Exception as e:
    print(f"   ❌ TorusCells import failed!")
    print(f"   Error: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n=== Diagnostic Complete ===")
