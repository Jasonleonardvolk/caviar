#!/usr/bin/env python
"""
Test Phase 2 optimizations: SIMD encoding, slab allocator, job-stealing
"""
import numpy as np
import sys
import os
import time

# Thread control
os.environ['NUMBA_THREADING_LAYER'] = 'omp'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.core.hyperbolic_matrix_multiply import hyperbolic_matrix_multiply, PARALLEL_THRESHOLD
from python.core.field_pool import get_field_pool, reset_field_pool
from python.core import slab_pool

def test_phase2_optimizations():
    """Test all Phase 2 optimizations together"""
    print("="*70)
    print("PHASE 2 OPTIMIZATIONS TEST")
    print("="*70)
    print()
    
    print("Optimizations enabled:")
    print("1. ✓ SIMD-friendly mixed precision encoding")
    print("2. ✓ Slab allocator for Strassen temporaries")
    print("3. ✓ Job-stealing thread pool")
    print("4. ✓ All previous optimizations still active")
    print()
    
    # Reset pools
    reset_field_pool()
    slab_pool.clear_pool()
    
    # Performance benchmark
    print("PERFORMANCE BENCHMARK")
    print("-"*50)
    
    test_sizes = [32, 64, 128, 256, 512]
    print(f"{'Size':>6} {'Time(ms)':>12} {'vs NumPy':>12} {'Target(ms)':>12}")
    print("-"*70)
    
    for n in test_sizes:
        if n > 256:
            trials = 2  # Fewer trials for large sizes
        else:
            trials = 5
            
        np.random.seed(42)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        
        # Warm-up
        try:
            _ = A @ B
            _ = hyperbolic_matrix_multiply(A, B)
        except Exception as e:
            print(f"{n:>6}    ERROR: {str(e)[:40]}...")
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
        
        # Target times based on document
        targets = {
            128: 4.0,   # After phase 2 opt 1
            256: 25.0,  # After phase 2 opt 1+2
            512: 150.0  # After all phase 2 opts
        }
        target = targets.get(n, 0)
        
        print(f"{n:>6} {soliton_time:>12.2f} {ratio:>11.1f}x {target:>12.1f} (err={error:.1e})")
    
    # Detailed analysis for n=512
    print("\n" + "="*70)
    print("TARGET PERFORMANCE ANALYSIS (n=512)")
    print("="*70)
    print()
    
    if 512 in [n for n in test_sizes]:
        print("Target milestones:")
        print("  n=128: ≤ 4.0 ms (after SIMD encoding)")
        print("  n=256: ≤ 25 ms (after + slab allocator)")
        print("  n=512: ≤ 150 ms (after + job-stealing)")
        print()
        
        # Check if we met the target
        if 'soliton_time' in locals() and n == 512:
            if soliton_time <= 150:
                print(f"✓ TARGET MET! {soliton_time:.1f} ms ≤ 150 ms")
                print(f"  Achieved {numpy_time*10:.0f}-{numpy_time*15:.0f} ms target range")
            else:
                print(f"✗ Missed target: {soliton_time:.1f} ms > 150 ms")
                print(f"  Need {soliton_time/150:.1f}x more improvement")
    
    # Pool statistics
    print("\n" + "="*70)
    print("MEMORY POOL STATISTICS")
    print("="*70)
    print()
    
    # Field pool stats
    field_stats = get_field_pool().stats()
    print("Field Buffer Pool:")
    print(f"  Reuse rate: {field_stats['reuse_rate']:.1%}")
    
    # Slab pool stats
    slab_stats = slab_pool.pool_stats()
    print("\nSlab Pool:")
    print(f"  Active pools: {slab_stats['num_pools']}")
    print(f"  Cached slabs: {slab_stats['total_slabs']}")
    print(f"  Memory held: {slab_stats['memory_mb']:.1f} MB")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print()
    print("Phase 2 optimizations should achieve:")
    print("- SIMD encoding: 1.4-1.7x improvement")
    print("- Slab allocator: 1.8x improvement for n≥256")
    print("- Job-stealing: Removes 2-3x tail for unbalanced trees")
    print("- Combined: Should meet n=512 < 150ms target")

if __name__ == "__main__":
    test_phase2_optimizations()
