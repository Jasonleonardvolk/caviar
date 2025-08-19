#!/usr/bin/env python
"""
Test the fixed SIMD encoding with proper scaling
"""
import numpy as np
import sys
import os

# Thread control
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
from python.core.field_pool import reset_field_pool

def test_fixed_simd():
    """Test fixed SIMD encoding"""
    print("="*70)
    print("TESTING FIXED SIMD ENCODING")
    print("="*70)
    print()
    
    reset_field_pool()
    
    print("Testing accuracy with fixed SIMD:")
    sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_expected = A @ B
        C_soliton = hyperbolic_matrix_multiply(A, B)
        
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        print(f"n={n:3d}: relative error = {error:.2e} {'✓' if error < 1e-10 else '✗'}")
        
        if error > 0.01:
            print(f"  WARNING: High error detected!")
            print(f"  First element - Expected: {C_expected[0,0]:.6f}, Got: {C_soliton[0,0]:.6f}")

if __name__ == "__main__":
    test_fixed_simd()
