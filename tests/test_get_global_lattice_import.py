#!/usr/bin/env python3
"""
Test script to verify get_global_lattice import and functionality
"""

# Test different possible import paths
print("Testing get_global_lattice import paths...")

try:
    from python.core.oscillator_lattice import get_global_lattice
    print("✓ Import from python.core.oscillator_lattice successful")
    lattice = get_global_lattice()
    print(f"  Lattice type: {type(lattice)}")
except ImportError as e:
    print(f"✗ Import from python.core.oscillator_lattice failed: {e}")

try:
    from oscillator_lattice import get_global_lattice
    print("✓ Import from oscillator_lattice successful")
except ImportError as e:
    print(f"✗ Import from oscillator_lattice failed: {e}")

try:
    from core.oscillator_lattice import get_global_lattice
    print("✓ Import from core.oscillator_lattice successful")
except ImportError as e:
    print(f"✗ Import from core.oscillator_lattice failed: {e}")

print("\nSuggested fix for imports in 03_fix_memory_fusion_fission_complete.py:")
print("Add at the top of the file:")
print("try:")
print("    from python.core.oscillator_lattice import get_global_lattice")
print("except ImportError:")
print("    from oscillator_lattice import get_global_lattice")
