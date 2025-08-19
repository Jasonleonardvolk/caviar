#!/usr/bin/env python
"""
Test with fixed Numba thread configuration
"""
import numpy as np
import sys
import os
import time

# CRITICAL: Set thread control BEFORE importing Numba
os.environ['NUMBA_THREADING_LAYER'] = 'omp'  # Use OpenMP, not TBB
os.environ['NUMBA_NUM_THREADS'] = '1'        # No parallelism to avoid conflicts
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply, PARALLEL_THRESHOLD
from python.core.field_pool import get_field_pool, reset_field_pool

def test_fixed_threads():
    """Test with proper thread configuration"""
    print("="*70)
    print("TESTING WITH FIXED THREAD CONFIGURATION")
    print("="*70)
    print()
    
    print("Thread configuration:")
    print("- NUMBA_THREADING_LAYER = omp (avoid TBB conflicts)")
    print("- NUMBA_NUM_THREADS = 1 (no nested parallelism)")
    print("- Removed parallel=True from Numba functions")
    print("- Disabled mixed precision temporarily")
    print()
    
    # Reset pool
    reset_field_pool()
    
    # Performance benchmark
    print("PERFORMANCE BENCHMARK")
    print("-"*50)
    
    test_sizes = [32, 64, 128, 256]
    baseline_times = {
        32: 0.06,    # From previous "all optimizations" test
        64: 0.36,
        128: 5.31,
        256: 44.71
    }
    
    print(f"{'Size':>6} {'Time(ms)':>12} {'Baseline(ms)':>12} {'Change':>10}")
    print("-"*60)
    
    for n in test_sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Warm-up
        _ = hyperbolic_matrix_multiply(A, B)
        
        # Time soliton
        times = []
        for _ in range(5):
            start = time.perf_counter()
            _ = hyperbolic_matrix_multiply(A, B)
            times.append(time.perf_counter() - start)
        
        soliton_time = np.median(times) * 1000
        baseline = baseline_times.get(n, 0)
        change = (soliton_time - baseline) / baseline * 100 if baseline > 0 else 0
        
        print(f"{n:>6} {soliton_time:>12.2f} {baseline:>12.2f} {change:>+9.1f}%")
    
    # Detailed analysis for key size
    print("\n" + "="*70)
    print("DETAILED TIMING BREAKDOWN (n=128)")
    print("="*70)
    print()
    
    n = 128
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    # Multiple runs to check consistency
    print("Individual trial times:")
    times = []
    for i in range(10):
        start = time.perf_counter()
        _ = hyperbolic_matrix_multiply(A, B)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
        if i < 5:
            print(f"  Trial {i+1}: {elapsed*1000:6.2f} ms")
    
    print(f"\nMedian: {np.median(times):.2f} ms")
    print(f"Min:    {np.min(times):.2f} ms")
    print(f"Max:    {np.max(times):.2f} ms")
    print(f"Std:    {np.std(times):.2f} ms")
    
    # Compare with target
    print("\n" + "="*70)
    print("PROGRESS TOWARD TARGET")
    print("="*70)
    print()
    
    print("Evolution of n=128 performance:")
    print("  Initial (broken parallel):     30.5 ms")
    print("  After threshold fix:            7.6 ms")
    print("  After quick wins:               5.3 ms")
    print("  With bad Numba+f32:            10.7 ms (regression)")
    print(f"  With fixes:                   ~{np.median(times):.1f} ms")
    
    # Estimate n=512
    if n == 128:
        # Strassen scaling
        est_512 = np.median(times) * (512/128)**2.807
        print(f"\nEstimated n=512: {est_512:.0f} ms")
        print(f"Target n=512:    150 ms")
        
        if est_512 < 150:
            print("✓ Should meet target!")
        else:
            print(f"Need {est_512/150:.1f}x more improvement")
            print("\nNext optimizations:")
            print("1. Re-enable mixed precision with proper SIMD")
            print("2. Slab allocator for late materialization")
            print("3. Job-stealing thread pool")

if __name__ == "__main__":
    test_fixed_threads()
