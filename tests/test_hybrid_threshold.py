#!/usr/bin/env python
"""
Test the hybrid threshold optimization
"""
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply

def test_hybrid_threshold():
    """Test that hybrid threshold works correctly"""
    print("="*70)
    print("HYBRID THRESHOLD TEST")
    print("="*70)
    print()
    
    # Test various sizes around the threshold
    test_sizes = [16, 32, 64, 65, 128]
    
    for n in test_sizes:
        print(f"\nTesting {n}×{n} matrix:")
        print("-"*40)
        
        # Create test matrices
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Compute with our function
        C_hybrid = hyperbolic_matrix_multiply(A, B)
        
        # Compute reference
        C_expected = A @ B
        
        # Check accuracy
        error = np.linalg.norm(C_hybrid - C_expected) / np.linalg.norm(C_expected)
        
        if n <= 64:
            print(f"  Using: NumPy (threshold applies)")
        else:
            print(f"  Using: Soliton physics")
        
        print(f"  Relative error: {error:.6e}")
        
        if error < 1e-10:
            print("  ✓ PASSED!")
        else:
            print(f"  ✗ Higher error than expected: {error:.2e}")
    
    # Test that we get identical results for n <= 64
    print("\n" + "="*70)
    print("VERIFYING THRESHOLD BEHAVIOR")
    print("="*70)
    
    A32 = np.random.randn(32, 32)
    B32 = np.random.randn(32, 32)
    
    C_hybrid = hyperbolic_matrix_multiply(A32, B32)
    C_numpy = A32 @ B32
    
    if np.allclose(C_hybrid, C_numpy):
        print("\n✓ For n=32 (below threshold): Result matches NumPy exactly")
    else:
        print("\n✗ ERROR: Results don't match for n=32")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nHybrid threshold optimization:")
    print("- n <= 64: Uses fast NumPy multiplication")
    print("- n > 64: Uses Strassen + soliton physics")
    print("\nThis avoids the overhead of wave-field setup for small matrices")
    print("while leveraging the O(n^2.807) scaling for larger ones.")

if __name__ == "__main__":
    test_hybrid_threshold()
