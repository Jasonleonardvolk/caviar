#!/usr/bin/env python
"""
Verify the optimized hybrid threshold (n=16)
"""
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply, HYBRID_THRESHOLD

def test_new_threshold():
    """Test the new threshold value"""
    print("="*70)
    print(f"VERIFYING NEW HYBRID THRESHOLD (n <= {HYBRID_THRESHOLD})")
    print("="*70)
    print()
    
    # Test sizes around the new threshold
    test_cases = [
        (8, "NumPy", True),
        (16, "NumPy", True),
        (17, "Soliton", False),
        (32, "Soliton", False),
        (64, "Soliton", False),
        (128, "Soliton", False),
    ]
    
    print(f"{'Size':>6} {'Expected':>10} {'Time(ms)':>10} {'Error':>12} {'Status':>10}")
    print("-"*60)
    
    for n, expected_method, uses_numpy in test_cases:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Time the operation
        start = time.perf_counter()
        C = hyperbolic_matrix_multiply(A, B)
        elapsed = (time.perf_counter() - start) * 1000
        
        # Check accuracy
        C_ref = A @ B
        error = np.linalg.norm(C - C_ref) / np.linalg.norm(C_ref)
        
        # Verify behavior
        if uses_numpy:
            status = "✓" if np.array_equal(C, C_ref) else "✗"
        else:
            status = "✓" if error < 1e-10 else "✗"
        
        print(f"{n:>6} {expected_method:>10} {elapsed:>10.3f} {error:>12.2e} {status:>10}")
    
    # Performance comparison at key sizes
    print("\n" + "="*70)
    print("PERFORMANCE IMPACT OF NEW THRESHOLD")
    print("="*70)
    print()
    
    sizes = [16, 32, 64, 128]
    print(f"{'Size':>6} {'NumPy(ms)':>12} {'Soliton(ms)':>12} {'Winner':>15}")
    print("-"*50)
    
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Time NumPy
        numpy_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = A @ B
            numpy_times.append(time.perf_counter() - start)
        numpy_time = np.median(numpy_times) * 1000
        
        # Time current implementation
        impl_times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = hyperbolic_matrix_multiply(A, B)
            impl_times.append(time.perf_counter() - start)
        impl_time = np.median(impl_times) * 1000
        
        if n <= HYBRID_THRESHOLD:
            winner = "NumPy (used)"
        else:
            winner = "NumPy" if numpy_time < impl_time else "Soliton"
        
        print(f"{n:>6} {numpy_time:>12.3f} {impl_time:>12.3f} {winner:>15}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nOptimized threshold: n <= {HYBRID_THRESHOLD}")
    print("\nBenefits:")
    print("- Avoids wave-field setup overhead for small matrices")
    print("- Leverages hand-tuned BLAS kernels where they excel")
    print("- Soliton physics kicks in only when n > 16")
    print("\nNext optimizations:")
    print("1. Batch collisions across quadrants")
    print("2. Reuse field buffers per recursion level")
    print("3. Parallel execution of Strassen products")

if __name__ == "__main__":
    test_new_threshold()
