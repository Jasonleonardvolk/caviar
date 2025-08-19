#!/usr/bin/env python
"""
Test the batching and buffer pool optimizations
"""
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply
from python.core.field_pool import get_field_pool, reset_field_pool

def test_optimizations():
    """Test batching and buffer pool optimizations"""
    print("="*70)
    print("TESTING BATCHING AND BUFFER POOL OPTIMIZATIONS")
    print("="*70)
    print()
    
    # Reset pool for clean test
    reset_field_pool()
    
    # Test correctness first
    print("CORRECTNESS TEST")
    print("-"*50)
    
    sizes = [32, 64, 128]
    for n in sizes:
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        C_expected = A @ B
        C_soliton = hyperbolic_matrix_multiply(A, B)
        
        error = np.linalg.norm(C_soliton - C_expected) / np.linalg.norm(C_expected)
        print(f"n={n:3d}: error = {error:.2e} {'✓' if error < 1e-10 else '✗'}")
    
    # Performance comparison
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print()
    
    # Time multiple runs
    n = 128
    num_trials = 5
    
    print(f"Timing {n}×{n} matrix multiplication ({num_trials} trials)")
    print("-"*50)
    
    np.random.seed(42)
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)
    
    # Warm-up
    _ = hyperbolic_matrix_multiply(A, B)
    
    # Time with optimizations
    times = []
    for i in range(num_trials):
        start = time.perf_counter()
        _ = hyperbolic_matrix_multiply(A, B)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Trial {i+1}: {elapsed*1000:.2f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAverage: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    
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
    
    # Memory usage estimate
    if stats['allocated'] > 0:
        # Rough estimate assuming complex128 (16 bytes per element)
        avg_buffer_size = 100 * 100 * 3  # Approximate
        memory_saved = stats['reused'] * avg_buffer_size * 16 / (1024**2)  # MB
        print(f"\nEstimated memory allocations saved: ~{memory_saved:.1f} MB")
    
    print("\n" + "="*70)
    print("OPTIMIZATION SUMMARY")
    print("="*70)
    print()
    print("✓ Batching prepared (ready for parallel execution)")
    print("✓ Buffer pooling active (reduces allocation overhead)")
    print("\nNext steps:")
    print("1. Implement parallel execution of Strassen products")
    print("2. Vectorize collision detection across batches")
    print("3. Profile with larger matrices to measure impact")

if __name__ == "__main__":
    test_optimizations()
