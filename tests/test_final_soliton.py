#!/usr/bin/env python
"""
Test the soliton physics implementation - FINAL VERSION
No monkey-patching needed since we fixed the core implementation
"""
import numpy as np
import sys
import os
from contextlib import contextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
from python.core import soliton_ring_sim, topology_catalogue

@contextmanager
def temp_collision_threshold(val):
    """Temporarily change collision threshold with guaranteed restoration"""
    orig = soliton_ring_sim._PEAK_ENERGY_THRESHOLD
    soliton_ring_sim._PEAK_ENERGY_THRESHOLD = val
    try:
        yield
    finally:
        soliton_ring_sim._PEAK_ENERGY_THRESHOLD = orig

def test_soliton_multiplication():
    """Test with the fixed implementation"""
    print("=" * 70)
    print("SOLITON PHYSICS TEST - FINAL VERSION")
    print("=" * 70)
    print()
    
    # Simple 2×2 test
    A = np.array([[1.0, 2.0], 
                  [3.0, 4.0]])
    B = np.array([[5.0, 6.0],
                  [7.0, 8.0]])
    
    print("Input matrices:")
    print("A =")
    print(A)
    print("\nB =")
    print(B)
    
    # Expected result
    C_expected = A @ B
    print("\nExpected C = A @ B =")
    print(C_expected)
    
    print("\n" + "-"*50)
    print("RUNNING SOLITON MULTIPLICATION")
    print("-"*50)
    
    try:
        # Run with default settings (fixed implementation should work)
        C_soliton = hyperbolic_matrix_multiply(A, B)
        print("\nSoliton result:")
        print(C_soliton)
        
        # Check accuracy
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        print(f"\nRelative error: {error:.6e}")
        
        if error < 0.01:  # 1% tolerance
            print("\n✓ Test PASSED! The soliton physics correctly computed the matrix product!")
        else:
            print(f"\n✗ Test FAILED - error {error:.2%} exceeds 1% threshold")
            
    except Exception as e:
        print(f"\n✗ Test FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
    
    # Analyze the mapping coverage
    print("\n" + "="*70)
    print("MAPPING COVERAGE ANALYSIS")
    print("="*70)
    
    proj = topology_catalogue.collision_projection(2)
    print(f"For 2×2 multiplication:")
    print(f"  Need: 2³ = 8 products (one for each (i,j,k) triple)")
    print(f"  Have: {len(proj._forward)} lattice positions mapped")
    
    if len(proj._forward) == 8:
        print("\n✓ All 8 mappings present!")
    else:
        print(f"\n✗ Missing {8 - len(proj._forward)} mappings")

def test_larger_matrices():
    """Test larger matrix sizes"""
    print("\n" + "="*70)
    print("TESTING LARGER MATRICES")
    print("="*70)
    
    for n in [3, 4]:
        print(f"\nTesting {n}×{n} matrices...")
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        try:
            C_soliton = hyperbolic_matrix_multiply(A, B)
            C_expected = A @ B
            error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
            
            if error < 0.01:
                print(f"  ✓ {n}×{n} PASSED (error: {error:.6e})")
            else:
                print(f"  ✗ {n}×{n} FAILED (error: {error:.6e})")
                
        except Exception as e:
            print(f"  ✗ {n}×{n} FAILED with exception: {e}")

if __name__ == "__main__":
    test_soliton_multiplication()
    test_larger_matrices()
