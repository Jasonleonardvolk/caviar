#!/usr/bin/env python
"""
Debug Phase 2 optimizations to find accuracy issue
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
from python.core.hyperbolic_matrix_multiply import _encode_matrices_to_wavefield
from python.core.topology_catalogue import collision_projection

def test_encoding_accuracy():
    """Test if mixed precision encoding is causing the error"""
    print("="*70)
    print("DEBUGGING ACCURACY ISSUE")
    print("="*70)
    print()
    
    # Test a simple 2x2 case first
    print("Testing 2x2 encoding:")
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    # Expected result
    C_expected = A @ B
    print(f"Expected:\n{C_expected}")
    
    # Test our soliton physics
    C_soliton = hyperbolic_matrix_multiply(A, B)
    print(f"Soliton result:\n{C_soliton}")
    print(f"Error: {np.linalg.norm(C_soliton - C_expected):.2e}")
    
    # Test larger sizes
    print("\nTesting accuracy at different sizes:")
    sizes = [4, 8, 16, 32, 64, 128]
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_expected = A @ B
        C_soliton = hyperbolic_matrix_multiply(A, B)
        
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        print(f"n={n:3d}: relative error = {error:.2e} {'✓' if error < 1e-10 else '✗'}")
        
        if error > 0.1:  # If error is huge
            print(f"  First element comparison:")
            print(f"  Expected: {C_expected[0,0]:.6f}")
            print(f"  Got:      {C_soliton[0,0]:.6f}")
            
            # Check encoding
            wavefield = _encode_matrices_to_wavefield(A[:2,:2], B[:2,:2])
            print(f"  Wavefield shape: {wavefield.shape}")
            print(f"  Wavefield dtype: {wavefield.dtype}")
            
            # Check if it's the SIMD encoding
            print(f"\n  Testing without SIMD encoding...")
            from python.core import simd_encode
            # Temporarily disable SIMD
            original_func = simd_encode.encode_with_mixed_precision
            simd_encode.encode_with_mixed_precision = None
            
            C_without_simd = hyperbolic_matrix_multiply(A, B)
            error_without = np.linalg.norm(C_without_simd - C_expected) / np.linalg.norm(C_expected)
            print(f"  Error without SIMD: {error_without:.2e}")
            
            # Restore
            simd_encode.encode_with_mixed_precision = original_func
            
            break

if __name__ == "__main__":
    test_encoding_accuracy()
