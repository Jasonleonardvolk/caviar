#!/usr/bin/env python
"""
Test to isolate the parallel execution issue at n=128
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

from python.core import hyperbolic_matrix_multiply as hmm
from python.core.field_pool import reset_field_pool

def test_parallel_issue():
    """Test parallel vs sequential at n=128"""
    print("="*70)
    print("ISOLATING PARALLEL EXECUTION ISSUE")
    print("="*70)
    print()
    
    reset_field_pool()
    
    # Test n=128 with different thresholds
    n = 128
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    C_expected = A @ B
    
    print(f"Current PARALLEL_THRESHOLD: {hmm.PARALLEL_THRESHOLD}")
    print()
    
    # Test 1: Normal execution (should use parallel)
    print("Test 1: Normal execution (uses parallel)")
    C1 = hmm.hyperbolic_matrix_multiply(A, B)
    error1 = np.linalg.norm(C1 - C_expected) / np.linalg.norm(C_expected)
    print(f"  Error: {error1:.2e} {'✓' if error1 < 1e-10 else '✗'}")
    
    # Test 2: Force sequential by temporarily raising threshold
    print("\nTest 2: Force sequential execution")
    original_threshold = hmm.PARALLEL_THRESHOLD
    hmm.PARALLEL_THRESHOLD = 256  # Force sequential for n=128
    
    C2 = hmm.hyperbolic_matrix_multiply(A, B)
    error2 = np.linalg.norm(C2 - C_expected) / np.linalg.norm(C_expected)
    print(f"  Error: {error2:.2e} {'✓' if error2 < 1e-10 else '✗'}")
    
    # Restore threshold
    hmm.PARALLEL_THRESHOLD = original_threshold
    
    # Test 3: Check if it's the slab pool
    print("\nTest 3: Clear slab pool and retry")
    from python.core import slab_pool
    slab_pool.clear_pool()
    reset_field_pool()
    
    C3 = hmm.hyperbolic_matrix_multiply(A, B)
    error3 = np.linalg.norm(C3 - C_expected) / np.linalg.norm(C_expected)
    print(f"  Error: {error3:.2e} {'✓' if error3 < 1e-10 else '✗'}")
    
    print("\nConclusion:")
    if error1 > 0.1 and error2 < 1e-10:
        print("✓ The issue is definitely in _strassen_parallel")
    elif error1 > 0.1 and error2 > 0.1:
        print("✗ The issue is NOT just parallel execution")
    
    # Test what strassen_parallel is doing
    if error1 > 0.1:
        print("\nDebugging _strassen_parallel:")
        print(f"  First element expected: {C_expected[0,0]:.6f}")
        print(f"  First element parallel: {C1[0,0]:.6f}")
        print(f"  First element sequential: {C2[0,0]:.6f}")

if __name__ == "__main__":
    test_parallel_issue()
