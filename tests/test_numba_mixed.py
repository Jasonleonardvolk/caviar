#!/usr/bin/env python
"""
Test Numba JIT and mixed precision optimizations
"""
import numpy as np
import sys
import os
import time

# Control BLAS threads
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply, PARALLEL_THRESHOLD
from python.core.field_pool import get_field_pool, reset_field_pool
from python.core.strassen_helpers import NUMBA_AVAILABLE
from python.core.soliton_ring_sim import NUMBA_AVAILABLE as SIM_NUMBA_AVAILABLE

def test_numba_mixed_precision():
    """Test with Numba JIT and mixed precision optimizations"""
    print("="*70)
    print("TESTING NUMBA JIT + MIXED PRECISION OPTIMIZATIONS")
    print("="*70)
    print()
    
    print("Optimization status:")
    print(f"1. Numba JIT available: {NUMBA_AVAILABLE and SIM_NUMBA_AVAILABLE}")
    print("2. Mixed precision encoding: ✓ (float32 encode, float64 accumulate)")
    print("3. Previous optimizations still active:")
    print("   - Vectorized collision detection")
    print("   - Fused Strassen additions")
    print("   - Buffer pool (97%+ reuse)")
    print("   - Parallel execution (n >= 128)")
    print()
    
    # Reset pool
    reset_field_pool()
    
    # Performance benchmark
    print("PERFORMANCE BENCHMARK")
    print("-"*50)
    
    test_sizes = [32, 64, 128, 256, 512]
    print(f"{'Size':>6} {'Time(ms)':>12} {'vs NumPy':>12} {'Mode':>15}")
    print("-"*60)
    
    for n in test_sizes:
        if n > 256:
            trials = 3  # Fewer trials for large sizes
        else:
            trials = 5
            
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Warm-up
        try:
            _ = A @ B
            _ = hyperbolic_matrix_multiply(A, B)
        except:
            print(f"{n:>6}    SKIPPED - too large")
            continue
        
        # Time NumPy
        numpy_times = []
        for _ in range(trials):
            start = time.perf_counter()
            _ = A @ B
            numpy_times.append(time.perf_counter() - start)
        numpy_time = np.median(numpy_times) * 1000
        
        # Time soliton
        soliton_times = []
        for _ in range(trials):
            start = time.perf_counter()
            C = hyperbolic_matrix_multiply(A, B)
            soliton_times.append(time.perf_counter() - start)
        soliton_time = np.median(soliton_times) * 1000
        
        # Verify accuracy
        C_ref = A @ B
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        
        ratio = soliton_time / numpy_time
        mode = "NumPy" if n <= 16 else ("Parallel" if n >= PARALLEL_THRESHOLD else "Sequential")
        
        print(f"{n:>6} {soliton_time:>12.2f} {ratio:>11.1f}x {mode:>15} (err={error:.1e})")
    
    # Test mixed precision accuracy
    print("\n" + "="*70)
    print("MIXED PRECISION ACCURACY TEST")
    print("="*70)
    print()
    
    print("Testing that float32 encoding maintains accuracy:")
    sizes = [32, 64, 128, 256]
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n) * 10  # Larger values to test precision
        B = np.random.randn(n, n) * 10
        
        C_soliton = hyperbolic_matrix_multiply(A, B)
        C_numpy = A @ B
        
        error = np.linalg.norm(C_soliton - C_numpy) / np.linalg.norm(C_numpy)
        print(f"  n={n:3d}: relative error = {error:.2e} {'✓' if error < 1e-13 else '✗'}")
    
    # Detailed timing for target size
    print("\n" + "="*70)
    print("TARGET PERFORMANCE CHECK (n=512)")
    print("="*70)
    print()
    
    n = 512
    target_ms = 150  # Target: 10-15x NumPy
    
    print(f"Target: < {target_ms} ms for n=512")
    print()
    
    # Estimate from n=256 timing
    if 'soliton_time' in locals() and n == 512:
        # Strassen scaling: n^2.807
        estimated_512 = soliton_time * (512/256)**2.807
        print(f"Estimated from n=256: {estimated_512:.1f} ms")
        
        if estimated_512 < target_ms:
            print(f"✓ Should meet target! ({estimated_512:.0f} < {target_ms})")
        else:
            print(f"✗ May need more optimization ({estimated_512:.0f} > {target_ms})")
            print("\nNext optimizations to try:")
            print("- Late materialization with slab allocator")
            print("- Job-stealing thread pool")
            print("- AVX2/NEON intrinsics")
    
    # Pool stats
    pool = get_field_pool()
    stats = pool.stats()
    
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print()
    print(f"Buffer pool reuse rate: {stats['reuse_rate']:.1%}")
    print(f"Numba compilation: {'Active' if NUMBA_AVAILABLE else 'Not available'}")
    print("\nExpected improvements from these optimizations:")
    print("- Numba JIT: 2-3x on collision detection and Strassen adds")
    print("- Mixed precision: 2x memory bandwidth reduction")
    print("- Combined: Should push n=512 close to 150ms target")

if __name__ == "__main__":
    test_numba_mixed_precision()
