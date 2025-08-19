#!/usr/bin/env python
"""
Test larger matrix sizes with timeout handling
"""
import numpy as np
import sys
import os
import time
import signal

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
from python.core import slab_pool

def test_large_sizes():
    """Test performance at target sizes"""
    print("="*70)
    print("TESTING LARGE MATRIX SIZES")
    print("="*70)
    print()
    
    reset_field_pool()
    slab_pool.clear_pool()
    
    # Test specific sizes
    sizes = [128, 256, 512]
    
    print(f"{'Size':>6} {'Soliton(ms)':>12} {'NumPy(ms)':>12} {'Ratio':>10} {'Target':>10}")
    print("-"*60)
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Time NumPy
        start = time.perf_counter()
        C_numpy = A @ B
        numpy_time = (time.perf_counter() - start) * 1000
        
        # Time soliton with timeout
        try:
            # Warm up
            if n <= 128:
                _ = hyperbolic_matrix_multiply(A[:16,:16], B[:16,:16])
            
            start = time.perf_counter()
            C_soliton = hyperbolic_matrix_multiply(A, B)
            soliton_time = (time.perf_counter() - start) * 1000
            
            # Verify accuracy
            error = np.linalg.norm(C_soliton - C_numpy) / np.linalg.norm(C_numpy)
            
            ratio = soliton_time / numpy_time
            target = 150 if n == 512 else (25 if n == 256 else 4)
            
            status = "✓" if soliton_time <= target else "✗"
            print(f"{n:6d} {soliton_time:12.1f} {numpy_time:12.2f} {ratio:9.1f}x {target:9.1f} {status}")
            
            if error > 1e-10:
                print(f"       WARNING: High error = {error:.2e}")
                
        except Exception as e:
            print(f"{n:6d}    ERROR: {str(e)[:40]}...")
            
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    # Performance projection
    if 'soliton_time' in locals() and n == 256:
        # Strassen scaling: n^2.807
        proj_512 = soliton_time * (512/256)**2.807
        print(f"\nProjected n=512 time: {proj_512:.0f} ms")
        print(f"Target: 150 ms")
        
        if proj_512 > 150:
            shortfall = proj_512 / 150
            print(f"\nNeed {shortfall:.1f}x more optimization to meet target")
            print("\nRemaining optimization opportunities:")
            print("1. GPU acceleration for collision detection")
            print("2. Cache-aware tiling for better memory access")
            print("3. SIMD intrinsics for critical loops")
            print("4. Reduce Python overhead with Cython")
        else:
            print("\n✓ Should meet target!")

if __name__ == "__main__":
    test_large_sizes()
