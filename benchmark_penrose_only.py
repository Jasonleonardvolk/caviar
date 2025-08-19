#!/usr/bin/env python3
"""
Benchmark ONLY Penrose topology - skip the slow O(n³) ones
"""
import os
import time
import numpy as np
import argparse

# Force experimental mode
os.environ["TORI_ENABLE_EXOTIC"] = "1"

from python.core.hot_swap_laplacian import HotSwappableLaplacian
from python.core.penrose_microkernel_v2 import multiply as penrose_multiply, RANK

def benchmark_penrose_only():
    sizes = [256, 512, 1024, 2048, 4096]
    RUNS_PER_SIZE = 7
    
    print(f"\nPENROSE-ONLY BENCHMARK (Rank={RANK})")
    print("=" * 60)
    print(f"Sizes: {sizes}")
    print(f"Runs per size: {RUNS_PER_SIZE}")
    print("=" * 60)
    
    # Initialize with Penrose
    hot_swap = HotSwappableLaplacian(
        initial_topology="penrose",
        lattice_size=(20, 20),
        enable_experimental=True,
    )
    
    print("\nBuilding Penrose Laplacian...")
    L = hot_swap.graph_laplacian
    
    times = []
    for n in sizes:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        
        # Warm-up
        _, _ = penrose_multiply(A, B, L)
        
        # Time multiple runs
        run_times = []
        for _ in range(RUNS_PER_SIZE):
            t0 = time.perf_counter()
            C, info = penrose_multiply(A, B, L)
            t1 = time.perf_counter()
            run_times.append(t1 - t0)
        
        median_time = np.median(run_times)
        times.append(median_time)
        
        print(f"  n={n}: {median_time*1000:.1f}ms (gap={info.get('spectral_gap', 0):.4f})")
    
    # Calculate omega
    log_sizes = np.log(sizes)
    log_times = np.log(times)
    slope, intercept = np.polyfit(log_sizes, log_times, 1)
    
    # R-squared
    residuals = log_times - (slope * log_sizes + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_times - np.mean(log_times))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\nRESULTS:")
    print(f"  penrose  ω ≈ {slope:.3f}  (R² = {r_squared:.3f})")
    print(f"           times: " + " ".join(f"{t*1000:6.1f}ms" for t in times))
    
    return slope, r_squared, times

if __name__ == "__main__":
    # Set single-threaded for consistency
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    omega, r2, times = benchmark_penrose_only()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"  Rank: {RANK}")
    print(f"  Omega: {omega:.4f}")
    print(f"  R²: {r2:.4f}")
    
    if omega < 2.371:
        print("\n  🎉 BELOW WORLD RECORD (2.371339)!")
    elif omega < 2.807:
        print("\n  ✓ Sub-Strassen achieved!")
    
    print("=" * 60)
