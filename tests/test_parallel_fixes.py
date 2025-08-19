#!/usr/bin/env python
"""
Test the parallel execution fixes with proper thread control
"""
import numpy as np
import sys
import os
import time

# Control BLAS threads to avoid oversubscription
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply, PARALLEL_THRESHOLD
from python.core.field_pool import get_field_pool, reset_field_pool

def test_parallel_fixes():
    """Test parallel execution with all fixes applied"""
    print("="*70)
    print("TESTING PARALLEL EXECUTION WITH FIXES")
    print("="*70)
    print()
    
    print("Environment settings:")
    print(f"- MKL_NUM_THREADS = 1 (avoid thread oversubscription)")
    print(f"- Parallel threshold raised to {PARALLEL_THRESHOLD}")
    print(f"- Buffer pool with context managers")
    print()
    
    # Reset pool for clean test
    reset_field_pool()
    
    # Test correctness
    print("CORRECTNESS TEST")
    print("-"*50)
    
    sizes = [32, 64, 128, 256]
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_expected = A @ B
        C_soliton = hyperbolic_matrix_multiply(A, B)
        
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        mode = "NumPy" if n <= 16 else ("Parallel" if n >= PARALLEL_THRESHOLD else "Sequential")
        print(f"n={n:3d}: error = {error:.2e} ({mode}) {'✓' if error < 1e-10 else '✗'}")
    
    # Performance comparison with different sizes
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print()
    
    test_sizes = [64, 128, 256]
    print(f"{'Size':>6} {'Time(ms)':>12} {'Mode':>15}")
    print("-"*40)
    
    for n in test_sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Warm-up
        _ = hyperbolic_matrix_multiply(A, B)
        
        # Time multiple runs
        times = []
        for _ in range(3):
            start = time.perf_counter()
            _ = hyperbolic_matrix_multiply(A, B)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times) * 1000
        mode = "Parallel" if n >= PARALLEL_THRESHOLD else "Sequential"
        print(f"{n:>6} {avg_time:>12.2f} {mode:>15}")
    
    # Check buffer pool stats
    pool = get_field_pool()
    stats = pool.stats()
    
    print("\n" + "="*70)
    print("BUFFER POOL STATISTICS")
    print("="*70)
    print()
    print(f"Buffers allocated: {stats['allocated']}")
    print(f"Buffers reused: {stats['reused']}")
    print(f"Reuse rate: {stats['reuse_rate']:.1%}")
    print(f"Currently pooled: {stats['pooled']}")
    
    if stats['reuse_rate'] > 0:
        print("\n✓ Buffer pool is working with context managers!")
    else:
        print("\n⚠ Buffer pool may still need debugging")
    
    # Compare with baseline NumPy
    print("\n" + "="*70)
    print("NUMPY BASELINE COMPARISON")
    print("="*70)
    print()
    
    n = 128
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    # Time NumPy
    numpy_times = []
    for _ in range(5):
        start = time.perf_counter()
        C_numpy = A @ B
        numpy_times.append(time.perf_counter() - start)
    numpy_avg = np.mean(numpy_times) * 1000
    
    # Time soliton
    soliton_times = []
    for _ in range(5):
        start = time.perf_counter()
        C_soliton = hyperbolic_matrix_multiply(A, B)
        soliton_times.append(time.perf_counter() - start)
    soliton_avg = np.mean(soliton_times) * 1000
    
    print(f"n=128 timing comparison:")
    print(f"  NumPy:   {numpy_avg:6.2f} ms")
    print(f"  Soliton: {soliton_avg:6.2f} ms")
    print(f"  Ratio:   {soliton_avg/numpy_avg:.1f}x slower")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Fixes applied:")
    print("✓ Raised PARALLEL_THRESHOLD to 128")
    print("✓ Added buffer pool context managers")
    print("✓ Set BLAS threads = 1 to avoid oversubscription")
    print("✓ Buffer pool in Strassen additions")
    print()
    print("Expected improvements:")
    print("- No more 7x slowdown from bad parallelization")
    print("- Buffer reuse should be > 0%")
    print("- Parallel should be faster than sequential for n>=128")

if __name__ == "__main__":
    test_parallel_fixes()
