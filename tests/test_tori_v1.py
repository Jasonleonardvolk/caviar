#!/usr/bin/env python
"""
Test the simplified TORI v1.0 implementation
"""
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply_v1 import hyperbolic_matrix_multiply

def test_tori_v1():
    """Test simplified TORI v1.0"""
    print("="*70)
    print("TORI v1.0 - SIMPLIFIED IMPLEMENTATION TEST")
    print("="*70)
    print()
    
    # Test correctness
    print("CORRECTNESS TEST")
    print("-"*50)
    sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_expected = A @ B
        C_tori = hyperbolic_matrix_multiply(A, B)
        
        error = np.linalg.norm(C_tori - C_expected) / np.linalg.norm(C_expected)
        print(f"n={n:3d}: relative error = {error:.2e} {'✓' if error < 1e-10 else '✗'}")
    
    # Performance benchmark
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    print()
    
    benchmark_sizes = [128, 256, 512]
    print(f"{'Size':>6} {'TORI(ms)':>12} {'NumPy(ms)':>12} {'Ratio':>10} {'Target(ms)':>12}")
    print("-"*60)
    
    for n in benchmark_sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Warm up
        _ = A @ B
        _ = hyperbolic_matrix_multiply(A, B)
        
        # Time NumPy
        trials = 5
        numpy_times = []
        for _ in range(trials):
            start = time.perf_counter()
            _ = A @ B
            numpy_times.append(time.perf_counter() - start)
        numpy_time = np.median(numpy_times) * 1000
        
        # Time TORI
        tori_times = []
        for _ in range(trials):
            start = time.perf_counter()
            _ = hyperbolic_matrix_multiply(A, B)
            tori_times.append(time.perf_counter() - start)
        tori_time = np.median(tori_times) * 1000
        
        ratio = tori_time / numpy_time
        target = {128: 4.0, 256: 25.0, 512: 150.0}.get(n, 0)
        
        status = "✓" if tori_time <= target else "✗"
        print(f"{n:6d} {tori_time:12.1f} {numpy_time:12.2f} {ratio:9.1f}x {target:11.1f} {status}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("TORI v1.0 Simplified Implementation:")
    print("✓ Removed expensive physics simulation")
    print("✓ Single fused kernel for 2×2 base case")
    print("✓ Numba JIT only on critical path")
    print("✓ Clean Strassen recursion")
    print("✓ NumPy threshold at n≤64")
    print()
    print("Expected performance:")
    print("- n=128: ~3ms (target 4ms)")
    print("- n=256: ~20ms (target 25ms)")
    print("- n=512: ~140ms (target 150ms)")

if __name__ == "__main__":
    test_tori_v1()
